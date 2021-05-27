from typing import Any, Dict, Tuple

from torchvision.datasets import MNIST

from vital.data.config import Tags


class MNIST(MNIST):
    def __getitem__(self, index: int) -> Dict[str, Any]:
        img, target = super().__getitem__(index)
        d = {Tags.img: img, Tags.gt: target}
        return d
