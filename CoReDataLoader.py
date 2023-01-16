from torch.utils.data import DataLoader, Dataset
import h5py as h5
from watpy.coredb.coredb import *
import pathlib as p
import sys
from os import devnull as print_supression_source
import numpy as np
import torch
from torch.nn.functional import pad as pad_tensor

maxlen = 4678
padlen = 4678 + 2


def rekey(inp_dict, keys_replace):
    return {keys_replace.get(k, k): v for k, v in inp_dict.items()}


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(print_supression_source, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class CoRe_DB_Dataset(Dataset):
    def __init__(
        self,
        sync=False,
        pad=True,
        cdb_path=p.Path("./CoRe_DB"),
        sel_attr=["id_eos", "id_mass_starA", "id_mass_starB"],
        device="cpu",
    ):
        self.sel_attr = sel_attr
        self.output_length = padlen
        self.sync = sync
        self.pad = pad
        self.path = cdb_path
        self.pspace = []
        self.eoss = []
        self.device = device
        with HiddenPrints():
            if not self.path.exists():
                self.path.mkdir(exist_ok=False)
            cdb = CoRe_db(self.path)
            if sync:
                cdb.sync(verbose=False, lfs=True)
            self.sims = cdb.sim
        for sim_key in self.sims:
            sim = self.sims[sim_key]
            for run_key in sim.run:
                run = sim.run[run_key]
                current_h5_filepath = p.Path(
                    self.sims[sim_key].run[run_key].data.path)
                self.eoss.append(run.md.data["id_eos"])
                current_h5_file = h5.File(current_h5_filepath / "data.h5", "r")
                current_rh_waveforms = [
                    i for i in current_h5_file.keys() if i == "rh_22"
                ]
                for selected_wf in current_rh_waveforms:
                    extraction_radii = list(
                        current_h5_file[selected_wf].keys())
                    tdata = current_h5_file[selected_wf][extraction_radii[0]][:, 1]
                    tdl = tdata[np.argmax(tdata):].shape[0]
                    if tdl < 500:
                        continue
                    for extraction_radius in extraction_radii:
                        self.pspace.append(
                            (sim_key, run_key, selected_wf, extraction_radius)
                        )
        self.pspace = tuple(self.pspace)
        self.eoss = list(sorted(set(self.eoss)))
        self.eosmap = {i: self.eoss.index(i) for i in self.eoss}
        self.numeoss = len(self.eoss)

    def __len__(self):
        return len(self.pspace)

    def __getitem__(self, index: int):
        return self.preprocess(self.retrieve(self.pspace[index]))

    def retrieve(self, psl):
        data = self.sims[psl[0]].run[psl[1]]
        h5path = p.Path(data.data.path) / "data.h5"
        metadata = {i: data.md.data[i] for i in self.sel_attr}
        data = h5.File(h5path, "r")[psl[2]][psl[3]][:, 1]
        data = data[np.argmax(data):: 4]
        data = data / np.linalg.norm(data)
        return data, metadata

    def preprocess(self, data):
        return self.preprocess_ts(data[0]).to(self.device), self.preprocess_params(
            data[1]
        ).to(self.device)

    def preprocess_ts(self, ts):
        ts = np.pad(ts, (int((maxlen - len(ts))/2), int((maxlen - len(ts))/2)),
                    "constant", constant_values=0)
        ts = np.pad(ts, (0, padlen - len(ts)), "constant", constant_values=0)
        return torch.tensor(ts).to(torch.float16)

    def preprocess_params(self, params):
        params = rekey(
            params,
            {
                "id_eos": "eos",
                "id_mass_starA": "m1s",
                "id_mass_starB": "m2s",
            },
        )
        params["eos"] = self.eosmap[params["eos"]]
        params["m1s"] = float(params["m1s"])
        params["m2s"] = float(params["m2s"])
        return torch.tensor(list(params.values())).to(torch.float32)


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = CoRe_DB_Dataset(sync=False, device=device)
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)
if __name__ == "__main__":
    print(next(iter(dataloader)))
