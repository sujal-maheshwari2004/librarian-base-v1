import math


def cosine_lr(step, config):

    if step < config.warmup_steps:
        return step / config.warmup_steps

    progress = (step - config.warmup_steps) / (
        config.total_steps - config.warmup_steps
    )

    cosine = 0.5 * (1 + math.cos(math.pi * progress))

    return config.min_lr / config.lr + cosine * (1 - config.min_lr / config.lr)