import time
import os
import torch
import requests
from tqdm import tqdm


GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Read once at import time — set DASHBOARD_KEY in your .env or shell
_DASHBOARD_KEY = os.environ.get("DASHBOARD_KEY", "")


class TrainingLogger:

    def __init__(self, seq_len, batch_size):

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.last_time = time.time()
        self.tokens_seen = 0

        self.run_id = int(time.time())

        self.api_endpoint = "https://librarian-logging-api-point.onrender.com/train_metrics"

        # Auth header sent with every POST — server doesn't gate /train_metrics
        # but having the key here means you can add ingest auth later with zero
        # changes to this file.
        self._headers = {
            "Content-Type": "application/json",
            **({"X-Dashboard-Key": _DASHBOARD_KEY} if _DASHBOARD_KEY else {}),
        }

        tqdm.write("────────────────────────────────────────")
        tqdm.write("RUN START")
        tqdm.write(f"run_id: {self.run_id}")
        tqdm.write(f"seq_len: {seq_len} | batch: {batch_size}")
        tqdm.write(f"logging key: {'set' if _DASHBOARD_KEY else 'NOT SET — metrics will still send'}")
        tqdm.write("────────────────────────────────────────")

    # ------------------------------------------------
    # Throughput
    # ------------------------------------------------
    def throughput(self):

        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now

        tokens = self.seq_len * self.batch_size
        self.tokens_seen += tokens

        if elapsed == 0:
            return 0

        return int(tokens / elapsed)

    # ------------------------------------------------
    # GPU Memory
    # ------------------------------------------------
    def gpu_mem(self):

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9

        return 0

    # ------------------------------------------------
    # Send
    # ------------------------------------------------
    def send_metrics(self, payload):

        try:
            requests.post(
                self.api_endpoint,
                json=payload,
                headers=self._headers,
                timeout=0.5
            )
        except Exception:
            pass

    # ------------------------------------------------
    # Train
    # ------------------------------------------------
    def train(self, step, loss, lr, grad):

        tok_s = self.throughput()
        gpu = self.gpu_mem()

        payload = {
            "run_id": self.run_id,
            "type": "train",
            "step": step,
            "loss": float(loss),
            "lr": float(lr),
            "grad_norm": float(grad),
            "tokens_per_sec": tok_s,
            "gpu_mem_gb": gpu,
            "timestamp": time.time()
        }

        self.send_metrics(payload)

        tqdm.write(
            f"{GREEN}TRAIN{RESET} "
            f"step {step:06d} | "
            f"loss {loss:6.4f} | "
            f"lr {lr:.2e} | "
            f"grad {grad:5.2f} | "
            f"tok/s {tok_s:6d} | "
            f"gpu {gpu:4.2f}GB"
        )

    # ------------------------------------------------
    # Eval
    # ------------------------------------------------
    def eval(self, step, val_loss):

        payload = {
            "run_id": self.run_id,
            "type": "eval",
            "step": step,
            "val_loss": float(val_loss),
            "timestamp": time.time()
        }

        self.send_metrics(payload)

        tqdm.write(
            f"{BLUE}EVAL {RESET} "
            f"step {step:06d} | "
            f"val_loss {val_loss:6.4f}"
        )

    # ------------------------------------------------
    # Checkpoint
    # ------------------------------------------------
    def checkpoint(self, step, val_loss):

        payload = {
            "run_id": self.run_id,
            "type": "checkpoint",
            "step": step,
            "val_loss": float(val_loss),
            "timestamp": time.time()
        }

        self.send_metrics(payload)

        tqdm.write(
            f"{YELLOW}CKPT {RESET} "
            f"step {step:06d} | "
            f"new_best {val_loss:6.4f}"
        )