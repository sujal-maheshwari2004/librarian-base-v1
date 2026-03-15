import time
import torch
from tqdm import tqdm


GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"


class TrainingLogger:

    def __init__(self, seq_len, batch_size):

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.last_time = time.time()
        self.tokens_seen = 0

        tqdm.write("────────────────────────────────────────")
        tqdm.write("RUN START")
        tqdm.write(f"seq_len: {seq_len} | batch: {batch_size}")
        tqdm.write("────────────────────────────────────────")

    def throughput(self):

        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now

        tokens = self.seq_len * self.batch_size
        self.tokens_seen += tokens

        if elapsed == 0:
            return 0

        return int(tokens / elapsed)

    def gpu_mem(self):

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9

        return 0

    def train(self, step, loss, lr, grad):

        tok_s = self.throughput()
        gpu = self.gpu_mem()

        tqdm.write(
            f"{GREEN}TRAIN{RESET} "
            f"step {step:06d} | "
            f"loss {loss:6.4f} | "
            f"lr {lr:.2e} | "
            f"grad {grad:5.2f} | "
            f"tok/s {tok_s:6d} | "
            f"gpu {gpu:4.2f}GB"
        )

    def eval(self, step, val_loss):

        tqdm.write(
            f"{BLUE}EVAL {RESET} "
            f"step {step:06d} | "
            f"val_loss {val_loss:6.4f}"
        )

    def checkpoint(self, step, val_loss):

        tqdm.write(
            f"{YELLOW}CKPT {RESET} "
            f"step {step:06d} | "
            f"new_best {val_loss:6.4f}"
        )