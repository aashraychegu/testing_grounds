#!/usr/bin/env python
# coding: utf-8

import pathlib as p
import sys
from typing import List, Union
from os import devnull as print_supression_source

import numpy as np
import torch
from torch.utils.data import *
from torch.nn.functional import pad as pad_tensor


def str_to_int(inp: str) -> int:
    out = 0
    for i in inp:
        if i.isdigit():
            out += int(i)
        else:
            out += ord(i)
    return out


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(print_supression_source, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class CoReDataSet(Dataset):
    def __init__(
        self,
        local_path: str = p.Path(__file__).parent.absolute()
        / "CoRe_DB_clone",  # type: ignore
        rh_xx: str = "rh_22",
        maxlen=40817,
        attrs: Union[List[str], str] = "*",
        gw_preprocess_func=True,
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
        self.maxlen = maxlen
        if gw_preprocess_func == True:
            self.gw_preprocessor = lambda x: pad_tensor(
                torch.from_numpy(x), (0, 0, 0, self.maxlen - x.shape[0]), "constant", 0
            )
        else:
            self.gw_preprocessor = gw_preprocess_func
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
        # return [
        #     self.gw_preprocessor(data.read(self.rh_xx)[:, 0:2]),
        #     {i: data.mdata.data[i]
        #         for i in data.mdata.data if i in self.attrs},
        # ]
        mdata = []
        for i in data.mdata.data:
            if i in self.attrs:
                if i == "id_eos":
                    mdata.append(str_to_int(data.mdata.data[i]))
                elif i == "database_key":
                    rstring = data.mdata.data[i]
                    sstring = rstring.split(":")
                    mdata.append(float(sstring[1]) * 10 + float(sstring[2][1:]))
                else:
                    mdata.append(float(data.mdata.data[i]))
        return self.gw_preprocessor(data.read(self.rh_xx)[:, 0:2]), torch.tensor(mdata)

    def collate_fn(self, data):
        return data


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
    c, batch_size=1, sampler=CoReSampler(c.indexes, len(c)), collate_fn=c.collate_fn
)

if __name__ == "__main__":
    print(next(iter(dataloader))[0][0].shape)
