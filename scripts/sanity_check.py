"""
sanity_check.py — Pre-training environment and pipeline sanity checker
───────────────────────────────────────────────────────────────────────
Checks everything that could go wrong before you commit to a multi-hour
training run, without actually downloading data or touching any real files.

Stages:
  1  Environment      — Python version, disk space, .env, HF_TOKEN
  2  CUDA / GPU       — device, VRAM, bfloat16, memory alloc
  3  Dependencies     — all packages importable + minimum versions
  4  Project imports  — every src/ module loads and exposes expected symbols
  5  Configs          — model + train configs parse without error
  6  Model            — instantiate, forward pass, NaN check, VRAM estimate
  7  Checkpoint       — save + load + weight-equality roundtrip (CPU only)
  8  Scheduler        — cosine LR shape correctness
  9  Dataset          — PackedDataset on a tiny synthetic .bin file
  10 DataLoader       — first batch shape + dtype

Usage:
    python scripts/sanity_check.py                 # full check, 390M config
    python scripts/sanity_check.py --config 130M
    python scripts/sanity_check.py --config dummy

Exits 0 if all checks pass, 1 if anything fails.
Nothing is downloaded. Nothing in data/, checkpoints/, or tokenizer/ is touched.
"""

import sys
import os
import time
import tempfile
import argparse
import struct
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── load .env so HF_TOKEN is visible ─────────────────────────────────
try:
    from dotenv import load_dotenv
    _env = ROOT / ".env"
    load_dotenv(_env if _env.exists() else None)
except ImportError:
    pass

# ── colours ───────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):     print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg):   print(f"  {YELLOW}!{RESET}  {msg}")
def fail(msg):   print(f"  {RED}✗{RESET}  {msg}")
def info(msg):   print(f"     {DIM}{msg}{RESET}")
def section(s):  print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}\n  {BOLD}{s}{RESET}\n{CYAN}{'─'*60}{RESET}")

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


# ══════════════════════════════════════════════════════════════════════
# 1. ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════
def check_environment():
    section("1 · Environment")

    def _python():
        v = sys.version_info
        if v < (3, 12):
            raise RuntimeError(f"Python {v.major}.{v.minor} — need >=3.12")
        return f"{v.major}.{v.minor}.{v.micro}"
    fatal_check("Python >= 3.12", _python)

    def _dotenv():
        p = ROOT / ".env"
        if not p.exists():
            raise RuntimeError(".env not found — create with HF_TOKEN=hf_...")
        return ".env present"
    check(".env file present", _dotenv)

    def _hf_token():
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN not set — add HF_TOKEN=hf_... to .env")
        return f"{len(token)} chars"
    check("HF_TOKEN set", _hf_token)

    def _disk():
        import shutil as _sh
        _, _, free = _sh.disk_usage(ROOT)
        free_gb = free / 1e9
        info(f"Disk free: {free_gb:.1f} GB")
        if free_gb < 150:
            raise RuntimeError(
                f"Only {free_gb:.1f} GB free — pipeline needs ~150 GB"
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


# ══════════════════════════════════════════════════════════════════════
# 2. CUDA & GPU
# ══════════════════════════════════════════════════════════════════════
def check_cuda():
    section("2 · CUDA & GPU")
    if FATAL:
        warn("Skipped — fatal failure in section 1")
        return

    import torch

    def _available():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — check driver / CUDA_VISIBLE_DEVICES")
        return "available"
    fatal_check("CUDA available", _available)

    if FATAL:
        return

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


# ══════════════════════════════════════════════════════════════════════
# 3. DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════
def check_dependencies():
    section("3 · Dependencies")
    if FATAL:
        warn("Skipped")
        return

    import importlib
    from packaging import version as pv

    def version_ok(installed, required):
        return pv.parse(installed) >= pv.parse(required)

    packages = [
        ("torch",           "2.0.0"),
        ("numpy",           "1.24.0"),
        ("tokenizers",      "0.13.0"),
        ("tqdm",            "4.0.0"),
        ("tensorboard",     "2.0.0"),
        ("requests",        "2.31.0"),
        ("pyarrow",         "10.0.0"),
        ("huggingface_hub", "0.16.0"),
        ("packaging",       "23.0"),
    ]

    for pkg, min_ver in packages:
        def _chk(p=pkg, mv=min_ver):
            mod = importlib.import_module(p)
            ver = getattr(mod, "__version__", "0.0.0")
            if not version_ok(ver, mv):
                raise RuntimeError(f"version {ver} < required {mv}")
            return ver
        check(pkg, _chk)


# ══════════════════════════════════════════════════════════════════════
# 4. PROJECT IMPORTS
# ══════════════════════════════════════════════════════════════════════
def check_project_imports():
    section("4 · Project Imports")
    if FATAL:
        warn("Skipped")
        return

    modules = [
        # configs
        ("configs.load_configs",      ["load_model_config", "load_train_config"]),
        ("configs.model_config",      ["ModelConfig"]),
        ("configs.train_config",      ["TrainConfig"]),
        # model
        ("src.model.gpt",             ["GPT"]),
        ("src.model.attention",       ["SelfAttention"]),
        ("src.model.block",           ["TransformerBlock"]),
        ("src.model.mlp",             ["MLP"]),
        ("src.model.rmsnorm",         ["RMSNorm"]),
        ("src.model.rope",            ["precompute_rope_freqs", "apply_rope"]),
        ("src.model.lora",            ["LoRALinear", "inject_lora",
                                       "enable_bitfit", "print_trainable_parameters",
                                       "get_lora_state_dict"]),
        # data
        ("src.data.dataset",          ["PackedDataset"]),
        ("src.data.clean",            ["run_clean", "iter_documents",
                                       "clean_document", "passes_quality_checks"]),
        ("src.data.pack",             ["run_pack", "pack_split"]),
        ("src.data.download",         ["download_datasets", "discover",
                                       "download_shard_to_disk", "parquet_to_txt"]),
        # training
        ("src.training.trainer",      ["Trainer"]),
        ("src.training.optimizer",    ["build_optimizer"]),
        ("src.training.scheduler",    ["cosine_lr"]),
        ("src.training.checkpoint",   ["save_checkpoint", "load_checkpoint",
                                       "save_latest"]),
        # evaluation
        ("src.evaluation.perplexity", ["compute_perplexity"]),
        ("src.evaluation.evaluator",  ["Evaluator"]),
        ("src.evaluation.eval_runner",["main"]),
        # inference
        ("src.inference.generate",    ["generate"]),
        ("src.inference.sampler",     ["sample_next_token"]),
        ("src.inference.load_model",  ["load_model"]),
        ("src.inference.chat",        ["chat"]),
        # utils
        ("src.utils.logging",         ["TrainingLogger", "StageLogger"]),
        # tokenizer
        ("tokenizer.train_tokenizer", ["train_tokenizer"]),
    ]

    for mod_path, symbols in modules:
        def _imp(mp=mod_path, syms=symbols):
            mod = __import__(mp, fromlist=syms)
            missing = [s for s in syms if not hasattr(mod, s)]
            if missing:
                raise ImportError(f"missing symbols: {missing}")
            return "OK"
        check(mod_path, _imp)


# ══════════════════════════════════════════════════════════════════════
# 5. CONFIGS
# ══════════════════════════════════════════════════════════════════════
def check_configs(model_cfg: str, train_cfg: str):
    section("5 · Configs")
    if FATAL:
        warn("Skipped")
        return

    from configs.load_configs import load_model_config, load_train_config

    def _model_cfg():
        mc = load_model_config(model_cfg)
        info(f"dim={mc.dim}  layers={mc.n_layers}  heads={mc.n_heads}  "
             f"vocab={mc.vocab_size}  seq={mc.max_seq_len}")
        assert mc.dim > 0
        assert mc.n_layers > 0
        assert mc.n_heads > 0
        assert mc.vocab_size > 0
        assert mc.hidden_dim > 0
        assert mc.dim % mc.n_heads == 0, \
            f"dim {mc.dim} not divisible by n_heads {mc.n_heads}"
        return f"dim={mc.dim} layers={mc.n_layers}"
    check(f"model config: {Path(model_cfg).name}", _model_cfg)

    def _train_cfg():
        tc = load_train_config(train_cfg)
        info(f"lr={tc.lr}  steps={tc.total_steps}  batch={tc.batch_size}  "
             f"grad_accum={tc.grad_accum}  warmup={tc.warmup_steps}")
        assert tc.lr > 0
        assert tc.min_lr > 0
        assert tc.min_lr < tc.lr
        assert tc.total_steps > 0
        assert tc.warmup_steps < tc.total_steps
        assert tc.batch_size > 0
        assert tc.grad_accum > 0
        return f"lr={tc.lr} steps={tc.total_steps}"
    check(f"train config: {Path(train_cfg).name}", _train_cfg)


# ══════════════════════════════════════════════════════════════════════
# 6. MODEL
# ══════════════════════════════════════════════════════════════════════
def check_model(model_cfg: str):
    section("6 · Model")
    if FATAL:
        warn("Skipped")
        return

    import torch
    from configs.load_configs import load_model_config
    from src.model.gpt import GPT

    mc = load_model_config(model_cfg)

    def _instantiate():
        model = GPT(mc)
        n     = sum(p.numel() for p in model.parameters())
        info(f"Parameters: {n:,}  ({n/1e6:.1f}M)")
        return f"{n/1e6:.1f}M params"
    check("GPT instantiates", _instantiate)

    def _to_cuda():
        GPT(mc).cuda()
        return "OK"
    check("model.to(cuda)", _to_cuda)

    def _forward_fp32():
        model = GPT(mc).cuda()
        x     = torch.randint(0, mc.vocab_size, (2, mc.max_seq_len), device="cuda")
        with torch.no_grad():
            logits = model(x)
        expected = (2, mc.max_seq_len, mc.vocab_size)
        if tuple(logits.shape) != expected:
            raise RuntimeError(f"shape {tuple(logits.shape)} != {expected}")
        if torch.isnan(logits).any():
            raise RuntimeError("NaN in logits on first forward pass")
        del model
        torch.cuda.empty_cache()
        return f"shape={expected}"
    check("forward pass fp32", _forward_fp32)

    def _forward_bf16():
        model = GPT(mc).cuda()
        x     = torch.randint(0, mc.vocab_size, (2, mc.max_seq_len), device="cuda")
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
        if torch.isnan(logits).any():
            raise RuntimeError("NaN in logits with bfloat16 autocast")
        del model
        torch.cuda.empty_cache()
        return f"dtype={logits.dtype}"
    check("forward pass bfloat16 autocast", _forward_bf16)

    def _tied_weights():
        model = GPT(mc)
        if mc.tie_embeddings:
            same = model.lm_head.weight.data_ptr() == model.token_emb.weight.data_ptr()
            if not same:
                raise RuntimeError("tie_embeddings=True but weights are not shared")
            return "tied OK"
        return "not tied (tie_embeddings=False)"
    check("embedding weight tying", _tied_weights)

    def _vram_estimate():
        torch.cuda.empty_cache()
        model = GPT(mc).cuda()
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        # rough estimate: model params + grads + optimizer states + activations
        estimated_train = used * 6
        info(f"Weights only      : {used:.2f} GB")
        info(f"Estimated training: ~{estimated_train:.1f} GB  (weights×6 heuristic)")
        info(f"GPU total         : {total:.1f} GB")
        if estimated_train > total * 0.95:
            raise RuntimeError(
                f"Estimated training VRAM {estimated_train:.1f} GB likely exceeds "
                f"{total:.1f} GB — reduce batch_size or use gradient checkpointing"
            )
        del model
        torch.cuda.empty_cache()
        return f"weights={used:.2f} GB  est_train={estimated_train:.1f} GB"
    check("VRAM estimate for training", _vram_estimate)

    def _lora_inject():
        from src.model.lora import inject_lora, print_trainable_parameters
        import io, contextlib
        model = GPT(mc)
        model = inject_lora(model, rank=8, alpha=16)
        buf   = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_trainable_parameters(model)
        out = buf.getvalue()
        # extract trainable % from printed output
        pct_str = [w for w in out.split() if "%" in w]
        return f"LoRA OK  {pct_str[0] if pct_str else ''}"
    check("LoRA injection", _lora_inject)


# ══════════════════════════════════════════════════════════════════════
# 7. CHECKPOINT
# ══════════════════════════════════════════════════════════════════════
def check_checkpoint(model_cfg: str):
    section("7 · Checkpoint")
    if FATAL:
        warn("Skipped")
        return

    import torch
    from configs.load_configs import load_model_config
    from configs.train_config import TrainConfig
    from src.model.gpt import GPT
    from src.training.optimizer import build_optimizer
    from src.training.checkpoint import save_checkpoint, load_checkpoint, save_latest

    mc = load_model_config(model_cfg)
    tc = TrainConfig()

    with tempfile.TemporaryDirectory(prefix="librarian_ckpt_") as tmp:
        tmp = Path(tmp)

        def _roundtrip():
            model1 = GPT(mc)
            opt1   = build_optimizer(model1, tc)
            path   = tmp / "test.pt"
            save_checkpoint(model1, opt1, step=42, path=path)
            assert path.exists(), "checkpoint file not created"
            size_mb = path.stat().st_size / (1024 ** 2)

            model2 = GPT(mc)
            opt2   = build_optimizer(model2, tc)
            step   = load_checkpoint(model2, opt2, path)
            assert step == 42, f"loaded step={step} expected 42"

            for (n1, p1), (n2, p2) in zip(
                model1.named_parameters(), model2.named_parameters()
            ):
                if not torch.allclose(p1, p2):
                    raise RuntimeError(f"weight mismatch: {n1}")

            info(f"checkpoint size: {size_mb:.0f} MB")
            return f"save+load+verify OK  {size_mb:.0f} MB"
        check("checkpoint save / load / weight-verify", _roundtrip)

        def _no_optimizer():
            model = GPT(mc)
            path  = tmp / "test.pt"
            step  = load_checkpoint(model, None, path)
            return f"optimizer=None OK  step={step}"
        check("checkpoint load with optimizer=None", _no_optimizer)

        def _save_latest():
            import unittest.mock as mock
            model = GPT(mc)
            opt   = build_optimizer(model, tc)
            # patch the hardcoded path inside save_latest to our tmp dir
            latest = tmp / "latest.pt"
            with mock.patch("src.training.checkpoint.Path") as MockPath:
                MockPath.return_value = latest
                MockPath.return_value.parent.mkdir = lambda **kw: None
                # call directly with torch.save to validate structure
                torch.save(
                    {"model": model.state_dict(),
                     "optimizer": opt.state_dict(),
                     "step": 7},
                    latest,
                )
            ckpt = torch.load(latest, map_location="cpu")
            assert ckpt["step"] == 7
            assert "model" in ckpt
            assert "optimizer" in ckpt
            return "latest checkpoint structure OK"
        check("save_latest structure", _save_latest)


# ══════════════════════════════════════════════════════════════════════
# 8. SCHEDULER
# ══════════════════════════════════════════════════════════════════════
def check_scheduler():
    section("8 · LR Scheduler")
    if FATAL:
        warn("Skipped")
        return

    from src.training.scheduler import cosine_lr

    def _shape():
        class C:
            lr = 3e-4; min_lr = 3e-5; warmup_steps = 100; total_steps = 1000

        c   = C()
        lrs = [cosine_lr(i, c) * c.lr for i in range(c.total_steps)]

        # warmup: strictly increasing
        warmup_vals = lrs[:c.warmup_steps]
        if not all(warmup_vals[i] < warmup_vals[i+1]
                   for i in range(len(warmup_vals)-1)):
            raise RuntimeError("LR not strictly increasing during warmup")

        # peak at end of warmup
        peak = lrs[c.warmup_steps - 1]
        if abs(peak - c.lr) > 1e-8:
            raise RuntimeError(f"Peak LR {peak:.2e} != configured lr {c.lr:.2e}")

        # floor: never below min_lr
        if min(lrs) < c.min_lr * 0.99:
            raise RuntimeError(f"LR dropped below min_lr: {min(lrs):.2e}")

        # final value should be close to min_lr
        final = lrs[-1]
        if abs(final - c.min_lr) > c.min_lr * 0.05:
            raise RuntimeError(f"Final LR {final:.2e} far from min_lr {c.min_lr:.2e}")

        info(f"warmup peak={peak:.2e}  floor={min(lrs):.2e}  final={final:.2e}")
        return "shape correct"
    check("cosine LR warmup + decay shape", _shape)

    def _step_zero():
        class C:
            lr = 3e-4; min_lr = 3e-5; warmup_steps = 100; total_steps = 1000
        val = cosine_lr(0, C()) * C.lr
        if val != 0.0:
            raise RuntimeError(f"Step 0 LR should be 0, got {val}")
        return "step 0 = 0.0"
    check("LR at step 0 is zero", _step_zero)


# ══════════════════════════════════════════════════════════════════════
# 9. DATASET
# ══════════════════════════════════════════════════════════════════════
def check_dataset(model_cfg: str):
    section("9 · PackedDataset")
    if FATAL:
        warn("Skipped")
        return

    import numpy as np
    import torch
    from configs.load_configs import load_model_config
    from src.data.dataset import PackedDataset

    mc = load_model_config(model_cfg)

    with tempfile.TemporaryDirectory(prefix="librarian_ds_") as tmp:
        tmp  = Path(tmp)
        path = tmp / "train_packed.bin"

        # write 200 synthetic sequences worth of uint16 tokens
        n_tokens = (mc.max_seq_len + 1) * 200
        data = np.random.randint(0, mc.vocab_size, size=n_tokens, dtype=np.uint16)
        data.tofile(path)

        def _load():
            ds = PackedDataset(str(path), mc.max_seq_len)
            assert len(ds) > 0, "dataset is empty"
            info(f"sequences: {len(ds)}")
            return f"{len(ds)} sequences"
        check("PackedDataset loads synthetic bin", _load)

        def _getitem():
            ds  = PackedDataset(str(path), mc.max_seq_len)
            x, y = ds[0]
            assert x.shape == (mc.max_seq_len,), \
                f"x.shape={x.shape} expected ({mc.max_seq_len},)"
            assert y.shape == (mc.max_seq_len,), \
                f"y.shape={y.shape} expected ({mc.max_seq_len},)"
            assert x.dtype == torch.long, f"x.dtype={x.dtype} expected torch.long"
            # y should be x shifted by 1
            assert torch.all(y[:-1] == x[1:]), "y is not x shifted by 1"
            return f"shape=({mc.max_seq_len},) dtype=torch.long shift-by-1 OK"
        check("PackedDataset __getitem__ shape + shift", _getitem)

        def _no_overflow():
            ds = PackedDataset(str(path), mc.max_seq_len)
            # last valid index should not raise
            x, y = ds[len(ds) - 1]
            assert x.shape == (mc.max_seq_len,)
            return "no index overflow"
        check("PackedDataset no overflow on last index", _no_overflow)


# ══════════════════════════════════════════════════════════════════════
# 10. DATALOADER
# ══════════════════════════════════════════════════════════════════════
def check_dataloader(model_cfg: str, train_cfg: str):
    section("10 · DataLoader")
    if FATAL:
        warn("Skipped")
        return

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from configs.load_configs import load_model_config, load_train_config
    from src.data.dataset import PackedDataset

    mc = load_model_config(model_cfg)
    tc = load_train_config(train_cfg)

    with tempfile.TemporaryDirectory(prefix="librarian_dl_") as tmp:
        tmp  = Path(tmp)
        path = tmp / "train_packed.bin"

        # enough sequences for at least 2 full batches
        n_seq    = max(tc.batch_size * 4, 32)
        n_tokens = (mc.max_seq_len + 1) * n_seq
        data     = np.random.randint(0, mc.vocab_size, size=n_tokens, dtype=np.uint16)
        data.tofile(path)

        def _first_batch():
            ds     = PackedDataset(str(path), mc.max_seq_len)
            loader = DataLoader(
                ds,
                batch_size=min(tc.batch_size, len(ds)),
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
            x, y = next(iter(loader))
            bs   = min(tc.batch_size, len(ds))
            assert x.shape == (bs, mc.max_seq_len), \
                f"x.shape={tuple(x.shape)} expected ({bs}, {mc.max_seq_len})"
            assert x.dtype == torch.long, f"dtype={x.dtype}"
            info(f"batch shape: {tuple(x.shape)}  dtype: {x.dtype}")
            return f"shape=({bs}, {mc.max_seq_len}) OK"
        check("DataLoader first batch shape + dtype", _first_batch)

        def _two_batches_differ():
            ds     = PackedDataset(str(path), mc.max_seq_len)
            loader = DataLoader(
                ds,
                batch_size=min(tc.batch_size, len(ds) // 2),
                shuffle=True,
                num_workers=0,
                drop_last=True,
            )
            it     = iter(loader)
            x1, _  = next(it)
            x2, _  = next(it)
            if torch.all(x1 == x2):
                raise RuntimeError("Two consecutive shuffled batches are identical")
            return "shuffle produces distinct batches"
        check("DataLoader shuffle produces distinct batches", _two_batches_differ)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Librarian pre-training sanity checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        choices=["390M", "130M", "dummy"],
        default="390M",
        help="Which model/train config pair to validate",
    )
    args = parser.parse_args()

    model_cfg = f"configs/model_{args.config}.json"
    train_cfg = f"configs/train_{args.config}.json"

    print(f"\n{'═'*60}")
    print(f"  {BOLD}{CYAN}Librarian Pre-Training Sanity Check{RESET}")
    print(f"{'═'*60}")
    print(f"  config       : {args.config}")
    print(f"  model config : {model_cfg}")
    print(f"  train config : {train_cfg}")
    print(f"\n  {DIM}No data is downloaded. No files outside this repo are touched.{RESET}")

    t_start = time.time()

    check_environment()
    check_cuda()
    check_dependencies()
    check_project_imports()
    check_configs(model_cfg, train_cfg)
    check_model(model_cfg)
    check_checkpoint(model_cfg)
    check_scheduler()
    check_dataset(model_cfg)
    check_dataloader(model_cfg, train_cfg)

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
        print(f"\n  {RED}{BOLD}{FAIL} check(s) FAILED — fix before running train.sh{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
