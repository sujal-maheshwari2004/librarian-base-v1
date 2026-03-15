import torch
from torch.utils.data import DataLoader

from configs.load_configs import load_model_config, load_train_config
from src.model.gpt import GPT
from src.data.dataset import PackedDataset
from src.training.trainer import Trainer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():

    model_config = load_model_config("configs/model.json")
    train_config = load_train_config("configs/train.json")

    model = GPT(model_config)

    train_dataset = PackedDataset(
        "data/tokenized/train_packed.bin",
        model_config.max_seq_len
    )

    val_dataset = PackedDataset(
        "data/tokenized/validation_packed.bin",
        model_config.max_seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        num_workers=4,
        pin_memory=True
    )

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        train_config,
        device=train_config.device
    )

    trainer.train()


if __name__ == "__main__":
    main()