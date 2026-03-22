"""
sanity_check.py — Full pre-training pipeline sanity checker
────────────────────────────────────────────────────────────
Works on a COMPLETELY FRESH server with NO existing data files.
Simulates the entire pipeline end-to-end in a temp directory using
a tiny slice of real data, so every burst point is exercised before
you commit to a real run.

Stages simulated (in order, matching train.sh):
  1  Environment      — Python, disk, HF_TOKEN, .env
  2  CUDA / GPU       — device, VRAM, bfloat16, memory alloc
  3  Dependencies     — all packages importable + versions
  4  Project imports  — every src/ module loads cleanly
  5  Download probe   — stream 200 rows from each HF dataset (no full download)
  6  Clean            — run clean pipeline on probe data → cleaned files
  7  Tokenizer train  — train a real BPE tokenizer on cleaned probe data
  8  Tokenize         — encode cleaned data → .bin files
  9  Pack             — pack .bin files into sequences
  10 Model            — instantiate, forward pass, VRAM, NaN checks
  11 Training loop    — 10-step mixed-precision train on packed probe data
  12 Checkpoint       — save + load + weight-equality roundtrip
  13 Evaluation       — perplexity on packed probe data

All pipeline stages write into an isolated temp directory that is
deleted on exit. Nothing touches data/, checkpoints/, or tokenizer/.

Usage:
    python sanity_check.py                   # full check, 390M config
    python sanity_check.py --config 130M
    python sanity_check.py --skip-pipeline   # skip stages 5-9
    python sanity_check.py --skip-e2e        # skip stages 10-13

Exits 0 if all pass, 1 if anything fails.
"""

import sys
import os
import time
import shutil
import argparse
import tempfile
from pathlib import Path

# ── project root on path ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── load .env immediately so HF_TOKEN is available everywhere ─
try:
    from dotenv import load_dotenv
    _env = ROOT / ".env"
    load_dotenv(_env if _env.exists() else None)
except ImportError:
    pass

# ── colours ───────────────────────────────────────────────────
GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED   = "\033[91m"
CYAN   = "\033[96m"; DIM    = "\033[2m";  RESET = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}!{RESET}  {msg}")
def fail(msg):  print(f"  {RED}✗{RESET}  {msg}")
def info(msg):  print(f"     {DIM}{msg}{RESET}")
def section(s): print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n  {BOLD}{s}{RESET}\n{CYAN}{'─'*60}{RESET}")

PASS  = 0
FAIL  = 0
FATAL = False


def check(name, fn):
    global PASS, FAIL
    try:
        result = fn()
        if result is False:
            fail(name)
            FAIL += 1
            return False
        label = f"  {DIM}({result}){RESET}" if isinstance(result, str) else ""
        ok(f"{name}{label}")
        PASS += 1
        return True
    except Exception as e:
        fail(f"{name}  →  {type(e).__name__}: {e}")
        FAIL += 1
        return False


def fatal_check(name, fn):
    global FATAL
    passed = check(name, fn)
    if not passed:
        FATAL = True
    return passed


# ══════════════════════════════════════════════════════════════
# 1. ENVIRONMENT
# ══════════════════════════════════════════════════════════════
def check_environment():
    section("1 · Environment")

    def _python():
        v = sys.version_info
        if v < (3, 12):
            raise RuntimeError(f"Python {v.major}.{v.minor} — need >=3.12")
        return f"{v.major}.{v.minor}.{v.micro}"
    fatal_check("Python >= 3.12", _python)

    def _dotenv_file():
        p = ROOT / ".env"
        if not p.exists():
            raise RuntimeError(".env not found — create with HF_TOKEN=hf_...")
        return ".env present"
    check(".env file present", _dotenv_file)

    def _hf_token():
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN not set — download will fail anonymously. "
                "Add HF_TOKEN=hf_... to .env"
            )
        return f"{len(token)} chars"
    check("HF_TOKEN set", _hf_token)

    def _disk():
        import shutil as _sh
        _, _, free = _sh.disk_usage(ROOT)
        free_gb = free / 1e9
        info(f"Disk free: {free_gb:.1f} GB")
        # raw datasets ~100GB + cleaned ~20GB + tokenized ~10GB + checkpoints
        if free_gb < 150:
            raise RuntimeError(
                f"Only {free_gb:.1f} GB free — pipeline needs ~150 GB "
                f"(raw + cleaned + tokenized + checkpoints)"
            )
        return f"{free_gb:.0f} GB free"
    check("Disk space >= 150 GB", _disk)

    def _cudnn():
        import torch
        if torch.cuda.is_available():
            info(f"cuDNN {torch.backends.cudnn.version()}  "
                 f"enabled={torch.backends.cudnn.enabled}")
        return "OK"
    check("cuDNN accessible", _cudnn)


# ══════════════════════════════════════════════════════════════
# 2. CUDA & GPU
# ══════════════════════════════════════════════════════════════
def check_cuda():
    section("2 · CUDA & GPU")
    if FATAL:
        warn("Skipped — fatal failure in section 1")
        return

    import torch

    def _available():
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch.cuda.is_available() = False — "
                "check CUDA_VISIBLE_DEVICES and driver"
            )
        return "available"
    fatal_check("CUDA available", _available)

    def _device_info():
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1e9
        info(f"Device  : {props.name}")
        info(f"VRAM    : {vram:.1f} GB")
        info(f"Compute : sm_{props.major}{props.minor}")
        if vram < 20:
            raise RuntimeError(f"Only {vram:.1f} GB VRAM — need >= 20 GB")
        return f"{vram:.1f} GB"
    check("VRAM >= 20 GB", _device_info)

    def _bf16():
        x = torch.ones(8, 8, dtype=torch.bfloat16, device="cuda")
        _ = x @ x
        return "bfloat16 matmul OK"
    check("bfloat16 matmul", _bf16)

    def _alloc_free():
        t = torch.empty(1024, 1024, 512, dtype=torch.float16, device="cuda")
        del t
        torch.cuda.empty_cache()
        return "2 GB alloc/free OK"
    check("2 GB CUDA alloc/free", _alloc_free)

    def _baseline():
        used = torch.cuda.memory_allocated() / 1e9
        free = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()) / 1e9
        info(f"Baseline: used={used:.2f} GB  free={free:.2f} GB")
        return f"{free:.1f} GB free"
    check("VRAM baseline", _baseline)


# ══════════════════════════════════════════════════════════════
# 3. DEPENDENCIES
# ══════════════════════════════════════════════════════════════
def check_dependencies():
    section("3 · Dependencies")
    if FATAL:
        warn("Skipped")
        return

    import importlib

    def vtuple(s):
        try:
            return tuple(int(x) for x in s.split(".")[:3])
        except Exception:
            return (0,)

    packages = [
        ("torch",       "2.0.0"),
        ("numpy",       "1.24.0"),
        ("datasets",    "2.0.0"),
        ("tokenizers",  "0.13.0"),
        ("tqdm",        "4.0.0"),
        ("tensorboard", "2.0.0"),
        ("requests",    "2.0.0"),
    ]

    for pkg, min_ver in packages:
        def _chk(p=pkg, mv=min_ver):
            mod = importlib.import_module(p)
            ver = getattr(mod, "__version__", "0.0.0")
            if vtuple(ver) < vtuple(mv):
                raise RuntimeError(f"version {ver} < required {mv}")
            return ver
        check(pkg, _chk)


# ══════════════════════════════════════════════════════════════
# 4. PROJECT IMPORTS
# ══════════════════════════════════════════════════════════════
def check_project_imports(model_cfg, train_cfg):
    section("4 · Project Imports")
    if FATAL:
        warn("Skipped")
        return

    modules = [
        ("configs.load_configs",           ["load_model_config", "load_train_config"]),
        ("src.model.gpt",                  ["GPT"]),
        ("src.model.attention",            ["SelfAttention"]),
        ("src.model.block",                ["TransformerBlock"]),
        ("src.model.mlp",                  ["MLP"]),
        ("src.model.rmsnorm",              ["RMSNorm"]),
        ("src.model.rope",                 ["precompute_rope_freqs", "apply_rope"]),
        ("src.model.lora",                 ["LoRALinear", "inject_lora"]),
        ("src.data.dataset",               ["PackedDataset"]),
        ("src.data.clean",                 ["run_clean"]),
        ("src.data.pack",                  ["run_pack"]),
        ("src.training.trainer",           ["Trainer"]),
        ("src.training.optimizer",         ["build_optimizer"]),
        ("src.training.scheduler",         ["cosine_lr"]),
        ("src.training.checkpoint",        ["save_checkpoint", "load_checkpoint"]),
        ("src.evaluation.perplexity",      ["compute_perplexity"]),
        ("src.utils.logging",              ["TrainingLogger", "StageLogger"]),
    ]

    for mod_path, symbols in modules:
        def _imp(mp=mod_path, syms=symbols):
            mod = __import__(mp, fromlist=syms)
            missing = [s for s in syms if not hasattr(mod, s)]
            if missing:
                raise ImportError(f"missing symbols: {missing}")
            return "OK"
        check(mod_path, _imp)

    def _configs():
        from configs.load_configs import load_model_config, load_train_config
        mc = load_model_config(model_cfg)
        tc = load_train_config(train_cfg)
        return (f"dim={mc.dim} layers={mc.n_layers} vocab={mc.vocab_size} | "
                f"lr={tc.lr} steps={tc.total_steps} batch={tc.batch_size}")
    check("configs parse correctly", _configs)


# ══════════════════════════════════════════════════════════════
# 5. DOWNLOAD PROBE  (200 rows per dataset — no full download)
# ══════════════════════════════════════════════════════════════
def check_download_probe(tmp: Path):
    section("5 · Download Probe  (200 rows per dataset)")
    if FATAL:
        warn("Skipped")
        return

    from datasets import load_dataset

    raw_dir = tmp / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    PROBE_ROWS = 200

    sources = [
        ("wikitext",    "Salesforce/wikitext", "wikitext-103-raw-v1", False),
        ("bookcorpus",  "rojagtap/bookcorpus",  None,                  False),
        ("openwebtext", "openwebtext",           None,                  False),
    ]

    for prefix, hf_path, hf_name, trust in sources:
        def _probe(pf=prefix, hp=hf_path, hn=hf_name, tr=trust):
            kwargs = dict(streaming=True, trust_remote_code=tr)
            if hn:
                kwargs["name"] = hn
            ds = load_dataset(hp, **kwargs)
            splits = list(ds.keys())
            total_rows = 0
            for split in splits:
                out = raw_dir / f"{pf}_{split}.txt"
                with out.open("w", encoding="utf-8") as f:
                    for i, ex in enumerate(ds[split]):
                        if i >= PROBE_ROWS:
                            break
                        text = (ex.get("text") or "").strip()
                        if text:
                            f.write(text + "\n\n")
                            total_rows += 1
            if total_rows == 0:
                raise RuntimeError(f"0 rows written — check HF_TOKEN and network")
            files = [p.name for p in raw_dir.glob(f"{pf}*.txt")]
            info(f"{pf}: {total_rows} rows  files={files}")
            return f"{total_rows} rows, {len(splits)} split(s)"
        check(f"probe: {prefix}", _probe)


# ══════════════════════════════════════════════════════════════
# 6. CLEAN STAGE
# ══════════════════════════════════════════════════════════════
def check_clean(tmp: Path):
    section("6 · Clean Stage")
    if FATAL:
        warn("Skipped")
        return

    import src.data.clean as clean_mod

    cleaned_dir = tmp / "data" / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp / "data" / "raw"

    orig_raw     = clean_mod.RAW_DIR
    orig_cleaned = clean_mod.CLEANED_DIR
    clean_mod.RAW_DIR     = raw_dir
    clean_mod.CLEANED_DIR = cleaned_dir

    try:
        def _run():
            summary = clean_mod.run_clean()
            merged = cleaned_dir / "merged_train.txt"
            if not merged.exists():
                raise RuntimeError("merged_train.txt not produced")
            lines = [l for l in merged.read_text(encoding="utf-8").splitlines() if l.strip()]
            if len(lines) == 0:
                raise RuntimeError(
                    "merged_train.txt is empty — all probe documents were filtered "
                    "by quality thresholds. This is expected with only 200 rows; "
                    "real pipeline will have enough data."
                )
            info(f"merged_train.txt: {len(lines)} lines")
            info(f"files_processed={summary.get('files_processed')}  "
                 f"docs_kept={summary.get('total_docs_kept')}")
            return f"{len(lines)} lines"
        check("clean + merge runs", _run)
    finally:
        clean_mod.RAW_DIR     = orig_raw
        clean_mod.CLEANED_DIR = orig_cleaned


# ══════════════════════════════════════════════════════════════
# 7. TOKENIZER TRAINING
# ══════════════════════════════════════════════════════════════
def check_tokenizer_train(tmp: Path):
    section("7 · Tokenizer Training")
    if FATAL:
        warn("Skipped")
        return

    import tokenizer.train_tokenizer as tok_mod

    tok_dir     = tmp / "tokenizer"
    cleaned_dir = tmp / "data" / "cleaned"
    tok_dir.mkdir(parents=True, exist_ok=True)

    orig_data  = tok_mod.DATA_FILE
    orig_dir   = tok_mod.TOKENIZER_DIR
    orig_vocab = tok_mod.VOCAB_SIZE
    tok_mod.DATA_FILE     = cleaned_dir / "merged_train.txt"
    tok_mod.TOKENIZER_DIR = tok_dir
    tok_mod.VOCAB_SIZE    = 2000   # tiny vocab — just proving the pipeline works

    try:
        def _train():
            summary = tok_mod.train_tokenizer()
            tok_json = tok_dir / "tokenizer.json"
            if not tok_json.exists():
                raise RuntimeError("tokenizer.json not produced")
            size_kb = tok_json.stat().st_size / 1024
            info(f"tokenizer.json: {size_kb:.0f} KB  "
                 f"vocab_size={summary.get('final_vocab_size')}")
            return f"vocab_size={summary.get('final_vocab_size')}"
        check("tokenizer trains on probe data", _train)
    finally:
        tok_mod.DATA_FILE     = orig_data
        tok_mod.TOKENIZER_DIR = orig_dir
        tok_mod.VOCAB_SIZE    = orig_vocab


# ══════════════════════════════════════════════════════════════
# 8. TOKENIZE STAGE
# ══════════════════════════════════════════════════════════════
def check_tokenize(tmp: Path):
    section("8 · Tokenization Stage")
    if FATAL:
        warn("Skipped")
        return

    import numpy as np
    import src.data.tokenizer as tok_stage

    tok_path    = tmp / "tokenizer" / "tokenizer.json"
    cleaned_dir = tmp / "data" / "cleaned"
    out_dir     = tmp / "data" / "tokenized"
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_tok = tok_stage.TOKENIZER_PATH
    orig_cln = tok_stage.CLEANED_DIR
    orig_out = tok_stage.TOKENIZED_DIR
    tok_stage.TOKENIZER_PATH = tok_path
    tok_stage.CLEANED_DIR    = cleaned_dir
    tok_stage.TOKENIZED_DIR  = out_dir

    try:
        def _tokenize():
            summary = tok_stage.run_tokenize()
            train_bin = out_dir / "train.bin"
            if not train_bin.exists():
                raise RuntimeError("train.bin not produced")
            data = np.fromfile(train_bin, dtype=np.uint16)
            info(f"train.bin: {len(data):,} tokens")
            if len(data) < 100:
                raise RuntimeError(
                    f"Only {len(data)} tokens in train.bin — "
                    "probe data too small, but pipeline itself works"
                )
            return f"{len(data):,} tokens in train.bin"
        check("tokenize pipeline runs", _tokenize)
    finally:
        tok_stage.TOKENIZER_PATH = orig_tok
        tok_stage.CLEANED_DIR    = orig_cln
        tok_stage.TOKENIZED_DIR  = orig_out


# ══════════════════════════════════════════════════════════════
# 9. PACK STAGE
# ══════════════════════════════════════════════════════════════
def check_pack(tmp: Path, model_cfg: str):
    section("9 · Pack Stage")
    if FATAL:
        warn("Skipped")
        return

    import numpy as np
    import src.data.pack as pack_mod
    from configs.load_configs import load_model_config

    mc      = load_model_config(model_cfg)
    tok_dir = tmp / "data" / "tokenized"

    orig_tok  = pack_mod.TOKENIZED_DIR
    orig_pack = pack_mod.PACKED_DIR
    orig_ctx  = pack_mod.CONTEXT_LENGTH
    pack_mod.TOKENIZED_DIR  = tok_dir
    pack_mod.PACKED_DIR     = tok_dir
    pack_mod.CONTEXT_LENGTH = mc.max_seq_len

    try:
        def _pack():
            pack_mod.run_pack()
            packed = tok_dir / "train_packed.bin"
            if not packed.exists():
                raise RuntimeError("train_packed.bin not produced")
            data  = np.memmap(packed, dtype=np.uint16, mode="r")
            n_seq = len(data) // mc.max_seq_len - 1
            info(f"train_packed.bin: {len(data):,} tokens  {n_seq} sequences of len {mc.max_seq_len}")
            if n_seq < 5:
                raise RuntimeError(
                    f"Only {n_seq} sequences — probe data too small to run "
                    f"DataLoader. Increase PROBE_ROWS in check_download_probe."
                )
            return f"{n_seq} sequences"
        check("pack pipeline runs", _pack)
    finally:
        pack_mod.TOKENIZED_DIR  = orig_tok
        pack_mod.PACKED_DIR     = orig_pack
        pack_mod.CONTEXT_LENGTH = orig_ctx


# ══════════════════════════════════════════════════════════════
# 10. MODEL
# ══════════════════════════════════════════════════════════════
def check_model(model_cfg: str):
    section("10 · Model")
    if FATAL:
        warn("Skipped")
        return

    import torch
    from configs.load_configs import load_model_config
    from src.model.gpt import GPT

    mc = load_model_config(model_cfg)

    def _instantiate():
        model = GPT(mc)
        n = sum(p.numel() for p in model.parameters())
        info(f"Parameters: {n:,}  ({n/1e6:.1f}M)")
        return f"{n/1e6:.1f}M params"
    check("GPT instantiates", _instantiate)

    def _to_cuda():
        GPT(mc).cuda()
        return "OK"
    check("model.to(cuda)", _to_cuda)

    def _forward_fp32():
        model = GPT(mc).cuda()
        x = torch.randint(0, mc.vocab_size, (2, mc.max_seq_len), device="cuda")
        with torch.no_grad():
            logits = model(x)
        expected = (2, mc.max_seq_len, mc.vocab_size)
        if logits.shape != expected:
            raise RuntimeError(f"shape {tuple(logits.shape)} != {expected}")
        if torch.isnan(logits).any():
            raise RuntimeError("NaN in logits on first forward pass")
        del model; torch.cuda.empty_cache()
        return f"shape={tuple(expected)}"
    check("forward pass fp32", _forward_fp32)

    def _forward_bf16():
        model = GPT(mc).cuda()
        x = torch.randint(0, mc.vocab_size, (2, mc.max_seq_len), device="cuda")
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
        if torch.isnan(logits).any():
            raise RuntimeError("NaN in logits with bfloat16 autocast")
        del model; torch.cuda.empty_cache()
        return f"dtype={logits.dtype}"
    check("forward pass bfloat16 autocast", _forward_bf16)

    def _vram_after_load():
        torch.cuda.empty_cache()
        model = GPT(mc).cuda()
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        info(f"VRAM with model loaded: {used:.2f} GB / {total:.1f} GB")
        del model; torch.cuda.empty_cache()
        return f"{used:.2f} GB"
    check("VRAM after model load", _vram_after_load)


# ══════════════════════════════════════════════════════════════
# 11. TRAINING LOOP
# ══════════════════════════════════════════════════════════════
def check_training_loop(tmp: Path, model_cfg: str, train_cfg: str):
    section("11 · Training Loop  (10 steps on probe data)")
    if FATAL:
        warn("Skipped")
        return

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from configs.load_configs import load_model_config, load_train_config
    from configs.train_config import TrainConfig
    from src.model.gpt import GPT
    from src.data.dataset import PackedDataset
    from src.training.optimizer import build_optimizer
    from src.training.scheduler import cosine_lr

    mc      = load_model_config(model_cfg)
    real_tc = load_train_config(train_cfg)

    # Tiny config — proves the loop works, doesn't consume real VRAM budget
    probe_tc = TrainConfig(
        batch_size=2,
        grad_accum=2,
        lr=3e-4,
        min_lr=3e-5,
        warmup_steps=3,
        total_steps=10,
        weight_decay=0.1,
        eval_interval=999,
        save_interval=999,
        mixed_precision=True,
        device="cuda",
    )

    packed_train = tmp / "data" / "tokenized" / "train_packed.bin"

    def _dataloader():
        ds     = PackedDataset(str(packed_train), mc.max_seq_len)
        loader = DataLoader(ds, batch_size=probe_tc.batch_size,
                            num_workers=2, pin_memory=True, drop_last=True)
        x, y = next(iter(loader))
        exp   = (probe_tc.batch_size, mc.max_seq_len)
        if tuple(x.shape) != exp:
            raise RuntimeError(f"x.shape={tuple(x.shape)} expected {exp}")
        if x.dtype != torch.long:
            raise RuntimeError(f"dtype={x.dtype} expected torch.long")
        return f"shape={exp} dtype={x.dtype}"
    check("DataLoader first batch", _dataloader)

    def _scheduler_shape():
        class C:
            lr=3e-4; min_lr=3e-5; warmup_steps=5; total_steps=100
        c   = C()
        lrs = [cosine_lr(i, c) * c.lr for i in range(100)]
        if lrs[0] >= lrs[5]:
            raise RuntimeError("LR not warming up")
        if min(lrs) < c.min_lr * 0.9:
            raise RuntimeError(f"LR dropped below min_lr: {min(lrs):.2e}")
        return f"peak={max(lrs):.2e}  floor={min(lrs):.2e}"
    check("cosine LR scheduler shape", _scheduler_shape)

    def _mini_train():
        torch.cuda.reset_peak_memory_stats()
        model  = GPT(mc).cuda()
        opt    = build_optimizer(model, probe_tc)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        ds     = PackedDataset(str(packed_train), mc.max_seq_len)
        loader = DataLoader(ds, batch_size=probe_tc.batch_size, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)
        model.train()
        losses = []
        step   = 0

        for x, y in loader:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss   = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1)
                ) / probe_tc.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % probe_tc.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                for g in opt.param_groups:
                    g["lr"] = probe_tc.lr * cosine_lr(step, probe_tc)
            losses.append(loss.item() * probe_tc.grad_accum)
            step += 1
            if step >= probe_tc.total_steps:
                break

        if any(torch.isnan(torch.tensor(l)) for l in losses):
            raise RuntimeError(f"NaN loss: {losses}")
        if losses[-1] > losses[0] * 5:
            raise RuntimeError(f"Loss exploded: {losses[0]:.4f} → {losses[-1]:.4f}")

        peak_vram  = torch.cuda.max_memory_allocated() / 1e9
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9

        # Project to real batch size
        scale      = real_tc.batch_size / probe_tc.batch_size
        proj_vram  = peak_vram * scale

        info(f"loss: {losses[0]:.4f} → {losses[-1]:.4f}")
        info(f"peak VRAM at batch={probe_tc.batch_size}: {peak_vram:.2f} GB")
        info(f"projected VRAM at real batch={real_tc.batch_size}: ~{proj_vram:.1f} GB / {total_vram:.1f} GB")

        if proj_vram > total_vram * 0.90:
            raise RuntimeError(
                f"Projected VRAM {proj_vram:.1f} GB > 90% of {total_vram:.1f} GB — "
                f"reduce batch_size in {train_cfg}"
            )

        del model; torch.cuda.empty_cache()
        return (f"10 steps OK  loss {losses[0]:.3f}→{losses[-1]:.3f}  "
                f"projected_vram={proj_vram:.1f}GB")
    check("10-step mixed-precision train", _mini_train)


# ══════════════════════════════════════════════════════════════
# 12. CHECKPOINT ROUNDTRIP
# ══════════════════════════════════════════════════════════════
def check_checkpoint(tmp: Path, model_cfg: str):
    section("12 · Checkpoint Save / Load")
    if FATAL:
        warn("Skipped")
        return

    import torch
    from configs.load_configs import load_model_config
    from configs.train_config import TrainConfig
    from src.model.gpt import GPT
    from src.training.optimizer import build_optimizer
    from src.training.checkpoint import save_checkpoint, load_checkpoint

    mc  = load_model_config(model_cfg)
    tc  = TrainConfig()
    ckp = tmp / "checkpoints"
    ckp.mkdir(parents=True, exist_ok=True)

    def _roundtrip():
        model1 = GPT(mc)
        opt1   = build_optimizer(model1, tc)
        path   = ckp / "sanity_test.pt"
        save_checkpoint(model1, opt1, step=99, path=path)
        if not path.exists():
            raise RuntimeError("checkpoint file not created")
        size_mb = path.stat().st_size / (1024**2)

        model2 = GPT(mc)
        opt2   = build_optimizer(model2, tc)
        step   = load_checkpoint(model2, opt2, path)
        if step != 99:
            raise RuntimeError(f"loaded step={step} expected 99")

        # Verify all weights match exactly
        for (n1, p1), (n2, p2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            if not torch.allclose(p1, p2):
                raise RuntimeError(f"weight mismatch after reload: {n1}")

        info(f"checkpoint size: {size_mb:.0f} MB")
        return f"save+load+weight-verify OK  {size_mb:.0f} MB"
    check("checkpoint save / load / weight-verify", _roundtrip)

    def _load_no_optimizer():
        from configs.load_configs import load_model_config
        model = GPT(load_model_config(model_cfg))
        path  = ckp / "sanity_test.pt"
        step  = load_checkpoint(model, None, path)  # optimizer=None must not crash
        return f"optimizer=None OK  step={step}"
    check("checkpoint load with optimizer=None", _load_no_optimizer)


# ══════════════════════════════════════════════════════════════
# 13. EVALUATION / PERPLEXITY
# ══════════════════════════════════════════════════════════════
def check_evaluation(tmp: Path, model_cfg: str):
    section("13 · Evaluation / Perplexity")
    if FATAL:
        warn("Skipped")
        return

    import torch
    from torch.utils.data import DataLoader
    from configs.load_configs import load_model_config
    from src.model.gpt import GPT
    from src.data.dataset import PackedDataset
    from src.evaluation.perplexity import compute_perplexity

    mc      = load_model_config(model_cfg)
    tok_dir = tmp / "data" / "tokenized"

    # Use validation_packed if available, else train_packed as proxy
    packed = (tok_dir / "validation_packed.bin"
              if (tok_dir / "validation_packed.bin").exists()
              else tok_dir / "train_packed.bin")

    def _ppl():
        model  = GPT(mc).cuda()
        ds     = PackedDataset(str(packed), mc.max_seq_len)
        loader = DataLoader(ds, batch_size=2, num_workers=0)
        ppl    = compute_perplexity(model, loader, "cuda")
        if torch.isnan(torch.tensor(ppl)):
            raise RuntimeError("perplexity is NaN")
        if ppl <= 0:
            raise RuntimeError(f"perplexity is non-positive: {ppl}")
        info(f"ppl={ppl:.2f}  (random-init model expected ≈ vocab_size={mc.vocab_size})")
        del model; torch.cuda.empty_cache()
        return f"ppl={ppl:.2f}"
    check("compute_perplexity runs cleanly", _ppl)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Librarian pre-training sanity checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",        choices=["390M", "130M", "dummy"],
                        default="390M")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip stages 5-9 (download → pack)")
    parser.add_argument("--skip-e2e",      action="store_true",
                        help="Skip stages 10-13 (model + training)")
    args = parser.parse_args()

    model_cfg = f"configs/model_{args.config}.json"
    train_cfg = f"configs/train_{args.config}.json"

    print(f"\n{'═'*60}")
    print(f"  {BOLD}{CYAN}Librarian Pre-Training Sanity Check{RESET}")
    print(f"{'═'*60}")
    print(f"  config        : {args.config}")
    print(f"  model config  : {model_cfg}")
    print(f"  train config  : {train_cfg}")
    print(f"  skip pipeline : {args.skip_pipeline}")
    print(f"  skip e2e      : {args.skip_e2e}")
    print(f"\n  {DIM}Stages 5-9 run in an isolated temp dir.")
    print(f"  Nothing is written to data/, checkpoints/, or tokenizer/.{RESET}")

    t_start = time.time()

    # ── always run ────────────────────────────────────────────
    check_environment()
    check_cuda()
    check_dependencies()
    check_project_imports(model_cfg, train_cfg)

    # ── pipeline simulation ───────────────────────────────────
    tmp = Path(tempfile.mkdtemp(prefix="librarian_sanity_"))
    info(f"\n  Temp dir: {tmp}")

    try:
        if not args.skip_pipeline:
            check_download_probe(tmp)
            check_clean(tmp)
            check_tokenizer_train(tmp)
            check_tokenize(tmp)
            check_pack(tmp, model_cfg)
        else:
            print(f"\n  {YELLOW}Stages 5–9 skipped (--skip-pipeline){RESET}")

        if not args.skip_e2e:
            packed = tmp / "data" / "tokenized" / "train_packed.bin"
            if not packed.exists():
                print(f"\n  {YELLOW}No probe packed data — "
                      f"stages 10-13 skipped (run without --skip-pipeline){RESET}")
            else:
                check_model(model_cfg)
                check_training_loop(tmp, model_cfg, train_cfg)
                check_checkpoint(tmp, model_cfg)
                check_evaluation(tmp, model_cfg)
        else:
            print(f"\n  {YELLOW}Stages 10–13 skipped (--skip-e2e){RESET}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        info("Temp dir cleaned up")

    # ── summary ───────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'═'*60}")
    print(f"  {BOLD}Results{RESET}")
    print(f"{'═'*60}")
    print(f"  {GREEN}Passed : {PASS}{RESET}")
    if FAIL > 0:
        print(f"  {RED}Failed : {FAIL}{RESET}")
    else:
        print(f"  {DIM}Failed : 0{RESET}")
    print(f"  Time   : {elapsed:.1f}s")
    print(f"{'═'*60}")

    if FAIL == 0:
        print(f"\n  {GREEN}{BOLD}ALL CHECKS PASSED — safe to run train.sh{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n  {RED}{BOLD}{FAIL} check(s) FAILED — do not run train.sh{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
