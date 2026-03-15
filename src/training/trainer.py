import torch
import torch.nn.functional as F
from tqdm import tqdm

from .optimizer import build_optimizer
from .scheduler import cosine_lr
from .checkpoint import save_checkpoint
from src.utils.logging import TrainingLogger


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        train_config,
        device
    ):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.config = train_config
        self.device = device

        self.optimizer = build_optimizer(model, train_config)

        self.step = 0
        self.best_val_loss = float("inf")

        self.logger = TrainingLogger(
            seq_len=model.config.max_seq_len,
            batch_size=train_config.batch_size
        )

        self.progress = tqdm(
            total=train_config.total_steps,
            desc="training",
            dynamic_ncols=True,
            leave=True
        )

    def grad_norm(self):

        total_norm = 0

        for p in self.model.parameters():

            if p.grad is None:
                continue

            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        return total_norm ** 0.5

    def train(self):

        model = self.model
        optimizer = self.optimizer
        config = self.config

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=config.mixed_precision
        )

        model.train()

        while self.step < config.total_steps:

            for batch in self.train_loader:

                x, y = batch

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.amp.autocast(
                    "cuda",
                    enabled=config.mixed_precision
                ):

                    logits = model(x)

                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1)
                    )

                    loss = loss / config.grad_accum

                scaler.scale(loss).backward()

                if (self.step + 1) % config.grad_accum == 0:

                    scaler.unscale_(optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0
                    )

                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad(set_to_none=True)

                    lr_mult = cosine_lr(self.step, config)

                    for g in optimizer.param_groups:
                        g["lr"] = config.lr * lr_mult

                # logging
                if self.step % 100 == 0:

                    lr = optimizer.param_groups[0]["lr"]

                    self.logger.train(
                        self.step,
                        loss.item(),
                        lr,
                        grad_norm
                    )

                # evaluation
                if self.step % config.eval_interval == 0 and self.step != 0:

                    val_loss = self.evaluate()

                    self.logger.eval(self.step, val_loss)

                    if val_loss < self.best_val_loss:

                        self.best_val_loss = val_loss

                        save_checkpoint(
                            model,
                            optimizer,
                            self.step,
                            f"checkpoints/best/best_{self.step}.pt"
                        )

                        self.logger.checkpoint(self.step, val_loss)

                self.step += 1
                self.progress.update(1)

                if self.step >= config.total_steps:
                    break

        self.progress.close()

    @torch.no_grad()
    def evaluate(self):

        self.model.eval()

        losses = []

        for batch in self.val_loader:

            x, y = batch

            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            losses.append(loss.item())

        self.model.train()

        return sum(losses) / len(losses)