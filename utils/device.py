import torch


def get_device():
    """Returns current torch device based on the availability of either CPU or GPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
