try:
    import torch
except Exception as e:
    print("Error: tensorflow not found")
    raise e
from .utils import mp_matrix
from .gcn import GCN, SageConv
