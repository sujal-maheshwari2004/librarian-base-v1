import torch
from torch.utils.data import DataLoader

from configs.load_configs import load_model_config, load_train_config
from src.model.gpt import GPT
from src.data.dataset import PackedDataset
from src.training.trainer import Trainer


def main():

    model_config = load_model_config("configs/model_config.json")
    train_config = load_train_config("configs/train_config.json")

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