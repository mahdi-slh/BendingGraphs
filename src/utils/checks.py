import torch
import numpy as np


def check_weights(params):
    """Checks weights for illegal values.

    Args:
        params (tensor): parameter tensor
    """
    hasNan = False
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any():
                print("NaN Values detected in model weight %s." % k)
                hasNan = True
            if torch.isinf(v).any():
                print("Nan Values detected in model weight %s." % k)
                hasNan = True
        elif isinstance(v, np.ndarray):
            if np.isnan(v).any():
                print("Inf Values detected in model weight %s." % k)
                hasNan = True
            if np.isinf(v).any():
                print("Inf Values detected in model weight %s." % k)
                hasNan = True
    return hasNan


def check_valid(*xx):
    """
    check if input is a valid value
    """
    hasNan = False
    for x in xx:
        if isinstance(x, torch.Tensor):
            if torch.isnan(x).any():
                print("NaN Values detected.")
                hasNan = True
            if torch.isinf(x).any():
                print("Inf Values detected.")
                hasNan = True
        elif isinstance(x, dict):
            hasNan |= check_weights(x)
        elif isinstance(x, np.ndarray):
            if np.isnan(x).any():
                print("NaN Values detected.")
                hasNan = True
            if np.isinf(x).any():
                print("Inf Values detected")
                hasNan = True
        elif isinstance(x, list):
            hasNan |= check_valid(*x)
        elif isinstance(x, str):
            pass
        else:
            print("unhandled type {}".format(type(x).__name__))
            hasNan = True
    return hasNan


def check_grads(*xx):
    """
    check if input is a valid value
    """
    for x in xx:
        if isinstance(x, dict):
            for k, v in x.items():
                print("Grad Values in  {}. {}".format(k, v.grad.view(-1)))
