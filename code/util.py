import torch, random, numpy as np

EPS = 1e-8

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)