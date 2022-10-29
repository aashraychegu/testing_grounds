#!/usr/bin/env python
# coding: utf-8

import os
import pathlib as p
import sys
from typing import List, Union

import torch
from torch.utils.data import *


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class CoReDataSet(Dataset):
    def __init__(
        self,
        local_path: str = p.Path(__file__).parent.absolute() / "CoRe_DB_clone",
        rh_xx: str = "rh_22",
        attrs: Union[List[str], str] = "*",
        gw_preprocess_func=lambda x: torch.from_numpy(x),
    ):
        self.local_path = p.Path(local_path)
        if not self.local_path.exists():
            self.local_path.mkdir(exist_ok=False)
        with HiddenPrints():
            from watpy.coredb.coredb import CoRe_db

            self.sim = CoRe_db(local_path).sim
        self.indexes = {}
        self.rh_xx = rh_xx
        self.attrs = attrs
        self.gw_preprocess_func = gw_preprocess_func
        index_tracker = 0
        for sim_key in self.sim:
            sel_sims = self.sim[sim_key]
            for sim_run in sel_sims.run:
                self.indexes[index_tracker] = f"{sim_key}:{sim_run}"
                index_tracker += 1
        self.count = len(self.indexes)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        if not idx in self.indexes.values():
            raise IndexError(f"the entered index is not valid {idx} {idx.split(':')}")
        split_idx = idx.split(":")
        sel_sim = split_idx[0] + ":" + split_idx[1]
        sel_run = split_idx[2]
        data = self.sim[sel_sim].run[sel_run].data
        if self.attrs == "*":
            self.attrs = list(data.mdata.data.keys())
        return self.gw_preprocess_func(data.read(self.rh_xx)), {
            i: data.mdata.data[i] for i in data.mdata.data if i in self.attrs
        }

    def collate_fn(self, data):
        return tuple(data)


class CoReSampler(Sampler):
    def __init__(self, indexes, ctr, generator=None):
        self.indexes = list(indexes.values())
        self.ctr = ctr
        self.generator = generator

    def __len__(self):
        return self.ctr

    def __iter__(self):
        for i in torch.randperm(len(self.indexes), generator=self.generator):
            yield self.indexes[i]


c = CoReDataSet(attrs=["id_eos", "id_mass_starA", "id_mass_starB", "database_key"])
dataloader = DataLoader(
    c, batch_size=64, sampler=CoReSampler(c.indexes, len(c)), collate_fn=c.collate_fn
)

if __name__ == "__main__":
    print(next(iter(dataloader))[0][1]["database_key"])
