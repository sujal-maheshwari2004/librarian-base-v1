import time
import os
import torch
import requests
from tqdm import tqdm


GREEN  = "\033[92m"
BLUE   = "\033[94m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
MAGENTA= "\033[95m"
RED    = "\033[91m"
DIM    = "\033[2m"
RESET  = "\033[0m"

_DASHBOARD_KEY = os.environ.get("DASHBOARD_KEY", "lib-450M-large")

# ────────────────────────────────────────────────────────────
# Stage definitions — order matters for the pipeline display
# ────────────────────────────────────────────────────────────
PIPELINE_STAGES = [
    "download",
    "clean",
    "train_tokenizer",
    "tokenize",
    "pack",
    "train",
    "evaluate",
]

STAGE_LABELS = {
    "download":        "Data Download",
    "clean":           "Data Cleaning",
    "train_tokenizer": "Tokenizer Training",
    "tokenize":        "Tokenization",
    "pack":            "Token Packing",
    "train":           "Model Training",
    "evaluate":        "Evaluation",
}

STAGE_ICONS = {
    "download":        "↓",
    "clean":           "✦",
    "train_tokenizer": "⌘",
    "tokenize":        "≋",
    "pack":            "▣",
    "train":           "◈",
    "evaluate":        "◉",
}


# ────────────────────────────────────────────────────────────
# Base sender — shared by both loggers
# ────────────────────────────────────────────────────────────
class _BaseSender:

    API_BASE = "https://librarian-logging-api-point.vercel.app"

    def __init__(self):
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_DASHBOARD_KEY}",
        }

    def _post(self, endpoint: str, payload: dict):
        try:
            r = requests.post(
                f"{self.API_BASE}/{endpoint}",
                json=payload,
                headers=self._headers,
                timeout=5,
            )
            if r.status_code not in (200, 201):
                tqdm.write(f"{RED}[logger] POST {endpoint} → {r.status_code}: {r.text}{RESET}")
        except Exception as e:
            tqdm.write(f"{RED}[logger] POST {endpoint} failed: {e}{RESET}")


# ────────────────────────────────────────────────────────────
# StageLogger — tracks pipeline stage progress + metrics
# ────────────────────────────────────────────────────────────
class StageLogger(_BaseSender):
    """
    Emit stage lifecycle events (start / progress / end / error)
    with optional metrics at each stage.

    Usage
    -----
        logger = StageLogger(run_id=run_id)

        logger.start("download")
        ...work...
        logger.progress("download", {"files_downloaded": 3, "bytes": 1_200_000})
        logger.end("download", {"total_files": 3, "total_bytes": 1_200_000})

        logger.start("train")
        ...
        logger.error("train", "CUDA OOM at step 500")
    """

    def __init__(self, run_id: int | None = None):
        super().__init__()
        self.run_id = run_id or int(time.time())
        self._stage_start: dict[str, float] = {}

        tqdm.write("────────────────────────────────────────")
        tqdm.write(f"{CYAN}PIPELINE{RESET}  run_id: {self.run_id}")
        tqdm.write("────────────────────────────────────────")

    # ── public API ──────────────────────────────────────────

    def start(self, stage: str):
        self._stage_start[stage] = time.time()
        label = STAGE_LABELS.get(stage, stage)
        icon  = STAGE_ICONS.get(stage, "›")

        tqdm.write(
            f"\n{CYAN}{icon} START   {RESET}{label}"
        )

        self._post("stage_metrics", {
            "run_id":    self.run_id,
            "stage":     stage,
            "event":     "start",
            "timestamp": time.time(),
            "metrics":   {},
        })

    def progress(self, stage: str, metrics: dict):
        label = STAGE_LABELS.get(stage, stage)
        icon  = STAGE_ICONS.get(stage, "›")

        parts = "  ".join(
            f"{k} {YELLOW}{_fmt_val(v)}{RESET}" for k, v in metrics.items()
        )
        tqdm.write(f"  {DIM}{icon}{RESET} {DIM}{label}{RESET}  {parts}")

        self._post("stage_metrics", {
            "run_id":    self.run_id,
            "stage":     stage,
            "event":     "progress",
            "timestamp": time.time(),
            "metrics":   _serialise(metrics),
        })

    def end(self, stage: str, metrics: dict | None = None):
        elapsed = time.time() - self._stage_start.get(stage, time.time())
        label   = STAGE_LABELS.get(stage, stage)
        icon    = STAGE_ICONS.get(stage, "›")
        metrics = metrics or {}

        parts = "  ".join(
            f"{k} {GREEN}{_fmt_val(v)}{RESET}" for k, v in metrics.items()
        )
        tqdm.write(
            f"  {GREEN}✓ DONE   {RESET}{label}  "
            f"{DIM}elapsed {elapsed:.1f}s{RESET}  {parts}"
        )

        self._post("stage_metrics", {
            "run_id":    self.run_id,
            "stage":     stage,
            "event":     "end",
            "elapsed_s": round(elapsed, 2),
            "timestamp": time.time(),
            "metrics":   _serialise(metrics),
        })

    def error(self, stage: str, message: str):
        label = STAGE_LABELS.get(stage, stage)
        tqdm.write(f"  {RED}✗ ERROR  {RESET}{label}  {message}")

        self._post("stage_metrics", {
            "run_id":    self.run_id,
            "stage":     stage,
            "event":     "error",
            "timestamp": time.time(),
            "metrics":   {"error": message},
        })


# ────────────────────────────────────────────────────────────
# TrainingLogger — unchanged API, now also accepts run_id
# ────────────────────────────────────────────────────────────
class TrainingLogger(_BaseSender):

    def __init__(self, seq_len: int, batch_size: int, run_id: int | None = None):
        super().__init__()

        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.run_id     = run_id or int(time.time())

        self.last_time   = time.time()
        self.tokens_seen = 0

        tqdm.write("────────────────────────────────────────")
        tqdm.write("RUN START")
        tqdm.write(f"run_id:  {self.run_id}")
        tqdm.write(f"seq_len: {seq_len}  batch: {batch_size}")
        tqdm.write(
            f"key:     {'set' if _DASHBOARD_KEY else 'NOT SET — metrics will still send'}"
        )
        tqdm.write("────────────────────────────────────────")

    # ── throughput ──────────────────────────────────────────
    def throughput(self):
        now     = time.time()
        elapsed = max(now - self.last_time, 1e-9)
        self.last_time = now
        tokens  = self.seq_len * self.batch_size
        self.tokens_seen += tokens
        return int(tokens / elapsed)

    # ── gpu mem ─────────────────────────────────────────────
    def gpu_mem(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0

    # ── send ────────────────────────────────────────────────
    def _send(self, payload: dict):
        self._post("train_metrics", payload)

    # ── train ───────────────────────────────────────────────
    def train(self, step, loss, lr, grad):
        tok_s = self.throughput()
        gpu   = self.gpu_mem()

        self._send({
            "run_id":          self.run_id,
            "type":            "train",
            "step":            step,
            "loss":            float(loss),
            "lr":              float(lr),
            "grad_norm":       float(grad),
            "tokens_per_sec":  tok_s,
            "gpu_mem_gb":      gpu,
            "timestamp":       time.time(),
        })

        tqdm.write(
            f"{GREEN}TRAIN{RESET} "
            f"step {step:06d} | "
            f"loss {loss:6.4f} | "
            f"lr {lr:.2e} | "
            f"grad {grad:5.2f} | "
            f"tok/s {tok_s:6d} | "
            f"gpu {gpu:4.2f}GB"
        )

    # ── eval ────────────────────────────────────────────────
    def eval(self, step, val_loss):
        self._send({
            "run_id":    self.run_id,
            "type":      "eval",
            "step":      step,
            "val_loss":  float(val_loss),
            "timestamp": time.time(),
        })

        tqdm.write(
            f"{BLUE}EVAL {RESET} "
            f"step {step:06d} | "
            f"val_loss {val_loss:6.4f}"
        )

    # ── checkpoint ──────────────────────────────────────────
    def checkpoint(self, step, val_loss):
        self._send({
            "run_id":    self.run_id,
            "type":      "checkpoint",
            "step":      step,
            "val_loss":  float(val_loss),
            "timestamp": time.time(),
        })

        tqdm.write(
            f"{YELLOW}CKPT {RESET} "
            f"step {step:06d} | "
            f"new_best {val_loss:6.4f}"
        )


# ────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────
def _fmt_val(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, int) and v > 1_000_000:
        return f"{v/1e6:.2f}M"
    if isinstance(v, int) and v > 1_000:
        return f"{v/1e3:.1f}K"
    return str(v)


def _serialise(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        else:
            out[k] = str(v)
    return out
