import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset
import copy
import random

from util.globals import *

REMOTE_ROOT = f"{REMOTE_ROOT_URL}/data/dsets"

class CounterFactDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        multi: bool = False,
        size: typing.Optional[int] = None,
        data_range: list = None,
        *args,
        **kwargs,
    ):
        data_dir = Path(data_dir)
        cf_loc = data_dir / "counterfact.json" 
        
        if not cf_loc.exists():
            assert False, f"Counterfact dataset not found at {cf_loc}"

        with open(cf_loc, "r") as f:
            self.data = json.load(f)
        if size is not None and data_range is not None:
            assert False, f"size and range can not be not None together"
        elif size is not None:
            self.data = self.data[:size]
        elif data_range is not None:
            self.data = [self.data[i] for i in data_range]

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class MultiCounterFactDataset(CounterFactDataset):
    def __init__(
        self, data_dir: str, size: typing.Optional[int] = None, *args, **kwargs
    ):
        super().__init__(data_dir, *args, multi=True, size=size, **kwargs)
