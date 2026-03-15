import torch
from pathlib import Path


def save_checkpoint(model, optimizer, step, path):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "step": step,
        },
        path
    )


def save_latest(model, optimizer, step):

    path = Path("checkpoints/latest.pt")
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "step": step,
        },
        path
    )


def load_checkpoint(model, optimizer, path):

    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model"])

    # FIXED: allow optimizer=None
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt.get("step", 0)
