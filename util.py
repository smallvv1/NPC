import os
import torch
import random
import numpy as np
import logging
import torch.backends.cudnn as cudnn
import pandas as pd

def set_seed_logger(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = False

    return args.seed

def get_logger(filename=None):
    logger = logging.getLogger('npc')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if filename is not None:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        absolute_filename = os.path.abspath(filename)
        has_file = any(
            isinstance(handler, logging.FileHandler)
            and os.path.abspath(handler.baseFilename) == absolute_filename
            for handler in logger.handlers
        )
        if not has_file:
            file_handler = logging.FileHandler(absolute_filename, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger
