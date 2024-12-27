import os
import pytz
import datetime
import random
import torch
import numpy as np


tensor2npy = lambda x: x.cpu().detach().numpy() if x.is_cuda else x.detach().numpy()
tensor2cpu = lambda x: x.cpu() if x.is_cuda else x

def set_same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # Mutil GPU environment
        torch.cuda.manual_seed_all(seed)
    # Cancel CuDNN opt
    torch.backends.cudnn.benchmark = False
    # Make CuDNN use same Algorithm
    torch.backends.cudnn.deterministic = True

class IDUtil(object):
    @staticmethod
    def get_random_id_bytime():
        tz = pytz.timezone('Asia/Shanghai')
        return datetime.datetime.now(tz).strftime("%Y-%m-%d-%H%M%S")

class PathUtil(object):
    @staticmethod
    def auto_create_folder_path(*args):
        for path in args:
            if not os.path.exists(path):
                os.makedirs(path)