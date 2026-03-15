import torch

from src.model.gpt import GPT


def load_model(model_config, checkpoint_path, device):

    model = GPT(model_config)

    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model"])

    model.to(device)

    model.eval()

    return model