import torch
import torch.nn.functional as F
from tqdm import tqdm

from .optimizer import build_optimizer
from .scheduler import cosine_lr
from .checkpoint import save_checkpoint


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

    def train(self):

        model = self.model
        optimizer = self.optimizer
        config = self.config

        model.train()

        scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

        for epoch in range(9999):

            for batch in tqdm(self.train_loader):

                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.cuda.amp.autocast(enabled=config.mixed_precision):

                    logits = model(x)

                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                    )

                    loss = loss / config.grad_accum

                scaler.scale(loss).backward()

                if (self.step + 1) % config.grad_accum == 0:

                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0
                    )

                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad()

                    lr_mult = cosine_lr(self.step, config)

                    for param_group in optimizer.param_groups:
                        param_group["lr"] = config.lr * lr_mult

                if self.step % 100 == 0:
                    print(f"step {self.step} | loss {loss.item():.4f}")

                if self.step % config.eval_interval == 0:
                    self.evaluate()

                if self.step % config.save_interval == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        self.step,
                        f"checkpoints/base/step_{self.step}.pt"
                    )

                self.step += 1

                if self.step >= config.total_steps:
                    return

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
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)

        print(f"\nValidation Loss: {avg_loss:.4f}\n")

        self.model.train()