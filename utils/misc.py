import os
import random
from datetime import datetime

import numpy as np
import torch


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def timestamp():
    return str(datetime.now())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
