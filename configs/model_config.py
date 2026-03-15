from dataclasses import dataclass


@dataclass
class ModelConfig:

    # tokenizer
    vocab_size: int = 16000

    # architecture
    dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    hidden_dim: int = 2048

    # sequence
    max_seq_len: int = 512

    # dropout
    dropout: float = 0.0

    # rope
    rope_theta: float = 10000.0

    # weight tying
    tie_embeddings: bool = True