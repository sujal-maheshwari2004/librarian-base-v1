import json
from .model_config import ModelConfig
from .train_config import TrainConfig


def load_model_config(path):

    with open(path) as f:
        data = json.load(f)

    return ModelConfig(**data)


def load_train_config(path):

    with open(path) as f:
        data = json.load(f)

    return TrainConfig(**data)