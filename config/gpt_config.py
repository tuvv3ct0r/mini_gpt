import torch

class GPTConfig:
    def __init__(self):
        self.vocab_size = None  # to be set after loading meta.pkl
        self.max_seq_len = 1024
        self.d_model = 512
        self.n_layers = 4
        self.n_heads = 4
        self.dropout = 0.1
        self.bias = True
        self.learning_rate = 3e-4
        self.batch_size = 16
        self.eval_interval = 500
        self.eval_iters = 200
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
