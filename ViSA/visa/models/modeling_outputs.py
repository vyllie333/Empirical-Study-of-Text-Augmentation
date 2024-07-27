from typing import Optional, Any, Tuple, List
from collections import OrderedDict

import torch

class ModelOutput(OrderedDict):
    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        return tuple(self[k] for k in self.keys())


class ABSAOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    t_loss: Optional[torch.FloatTensor] = None
    d_loss: Optional[torch.FloatTensor] = None
    aspects: Optional[List[int]] = None
    polarities: Optional[List[int]] = None
