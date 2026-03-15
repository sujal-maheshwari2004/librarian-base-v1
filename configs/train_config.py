from dataclasses import dataclass


@dataclass
class TrainConfig:

    batch_size: int = 32
    grad_accum: int = 4

    lr: float = 3e-4
    min_lr: float = 3e-5

    warmup_steps: int = 1000
    total_steps: int = 50000

    weight_decay: float = 0.1

    eval_interval: int = 1000
    save_interval: int = 5000

    mixed_precision: bool = True

    device: str = "cuda"