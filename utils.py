import os
import copy
import torch
import random
import logging
import numpy as np
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR
from collections import OrderedDict, deque
from mmcv.runner.optimizer import OPTIMIZER_BUILDERS as MMCV_OPTIMIZER_BUILDERS
from mmcv.utils import Registry, build_from_cfg


OPTIMIZER_BUILDERS = Registry(
    'optimizer builder', parent=MMCV_OPTIMIZER_BUILDERS)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def set_random_seed(seed):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pretrained_model(weight_path, model):
    checkpoint = torch.load(weight_path, map_location='cpu')
    if 'model' in checkpoint:
        static_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        static_dict = checkpoint['state_dict']
    else:
        raise ValueError('None state_dict in checkpoint file')
    new_state_dict = OrderedDict()
    for k, v in static_dict.items():
        head = k.split('.')[0]
        if head == 'backbone':
            new_state_dict[k] = v
        elif head == 'transformer':
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)

    return model


def save_model(model, out_dir, filename='best.pth'):
    """Save the state dict of model
    """
    torch.save({'model': model.state_dict()}, os.path.join(out_dir, filename))


def save_checkpoint(model, optimizer, lr_scheduler, scaler, meta, out_dir):
    """Save the checkpoint with mmlab template
    """
    torch.save(dict(model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    lr_scheduler=lr_scheduler.state_dict(),
                    scaler=scaler.state_dict() if scaler else None,
                    meta=meta),
               os.path.join(out_dir, 'checkpoint.pth'))


def get_logger(name, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    logger.propagate = False

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='w', delay=True)
        handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    return logger


def build_optimizer(model, cfg):
    kwargs = cfg.copy()
    if 'lr_decay' in kwargs:
        lr_decay = kwargs.pop('lr_decay')

        param_dict = [{'params': []}]
        for item in lr_decay:
            param_dict.append({
                'params': [],
                'lr': kwargs['lr'] * lr_decay[item]
            })

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            idx = 0
            for i, item in enumerate(lr_decay):
                if item in n:
                    idx = i + 1
                    break
            param_dict[idx]['params'].append(p)
    else:
        param_dict = model.parameters()

    kwargs['params'] = param_dict
    type = kwargs.pop('type')
    module = getattr(torch.optim, type)

    return module(**kwargs)


class SmoothedLossContainer(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, log_key, log_interval=10):
        self.log_key = log_key
        self.log_interval = log_interval
        self.deque_dict = dict()
        for k in self.log_key:
            self.deque_dict[k] = deque(maxlen=log_interval)

    def update(self, losses):
        for k in self.log_key:
            assert k in losses
            if is_dist_avail_and_initialized():
                loss_k = losses[k].clone().detach()
                loss_value = loss_k / get_world_size()
                torch.distributed.all_reduce(loss_value)
            else:
                loss_value = losses[k]
            self.deque_dict[k].append(loss_value.item())

    def fetch_mean(self):
        values_dict = dict()
        for k in self.log_key:
            data_arr = np.array(list(self.deque_dict[k]))
            values_dict[k] = data_arr.mean()

        return values_dict


class WarmupMultiStepLR(MultiStepLR):
    r"""
    # max_iter = epochs * steps_per_epoch
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters (int): The total number of warm-up iterations.
        milestones (list)  List of iter indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay. Default: 0.1.
        warmup_ratio (float):Default: 0.001.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer, milestones, warmup_iters, warmup_ratio=1e-3,  gamma=0.1, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        if self._step_count <= self.warmup_iters:
            alpha = (1 - self.warmup_ratio) / (self.warmup_iters - 1)
            warmup_factor = alpha * (self._step_count - 1) + self.warmup_ratio
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            lr = super().get_lr()
        return lr

