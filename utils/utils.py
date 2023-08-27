import argparse
import os
import sys
import time
import warnings
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def str2bool(v: str, strict=True) -> bool:
    if isinstance(v, bool):
        return v
    elif isinstance(v, str):
        if v.lower() in ("true", "yes", "on" "t", "y", "1"):
            return True
        elif v.lower() in ("false", "no", "off", "f", "n", "0"):
            return False
    if strict:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")
    else:
        return True


def to_cuda(data, device="cuda", exclude_keys: "list[str]" = None):
    if isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, (tuple, list, set)):
        data = [to_cuda(b, device) for b in data]
    elif isinstance(data, dict):
        if exclude_keys is None:
            exclude_keys = []
        for k in data.keys():
            if k not in exclude_keys:
                data[k] = to_cuda(data[k], device)
    else:
        # raise TypeError(f"Unsupported type: {type(data)}")
        data = data
    return data


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = "w"
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if "\r" in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def get_network(arch: str, isTrain=False, continue_train=False, init_gain=0.02, pretrained=True):
    if "resnet" in arch:
        from networks.resnet import ResNet

        resnet = getattr(import_module("networks.resnet"), arch)
        if isTrain:
            if continue_train:
                model: ResNet = resnet(num_classes=1)
            else:
                model: ResNet = resnet(pretrained=pretrained)
                model.fc = nn.Linear(2048, 1)
                nn.init.normal_(model.fc.weight.data, 0.0, init_gain)
        else:
            model: ResNet = resnet(num_classes=1)
        return model
    else:
        raise ValueError(f"Unsupported arch: {arch}")


def pad_img_to_square(img: np.ndarray):
    H, W = img.shape[:2]
    if H != W:
        new_size = max(H, W)
        img = np.pad(img, ((0, new_size - H), (0, new_size - W), (0, 0)), mode="constant")
        assert img.shape[0] == img.shape[1] == new_size
    return img
