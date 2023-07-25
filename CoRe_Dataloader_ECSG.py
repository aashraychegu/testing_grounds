from torch.utils.data import DataLoader, Dataset, TensorDataset
import h5py as h5
from watpy.coredb.coredb import *
import pathlib as p
import sys
from os import devnull as print_supression_source
import numpy as np
import torch
from scipy.signal import argrelextrema
import pywt
import random
import math
import time

scale_min = 1
scale_max = 201
dscale = 0.1
FIN_WIDTH = 400
TOLERACE = 400
pad_to_for_planck_window = 90
p2 = pad_to_for_planck_window


def planck_window(j: int, N: int):
    window = np.linspace(0, j - 1, j - 1)
    window[0] = 1
    window = 1.0 / (1.0 + np.exp(j / window - j / (j - window)))
    window[0] = 0
    window = np.concatenate(
        (
            window,
            np.ones((N - (j * 2))),
            np.flip(
                window,
            ),
        )
    )
    return window


def cut_at_lowest_envelope(hplus, hcross):
    # Cutting inspiral off
    oenv = np.sqrt(hplus**2 + hcross**2)
    cut_point = np.argmax(hplus)
    mhplus = hplus[cut_point:]
    env = oenv[cut_point:]
    envcut = argrelextrema(env, np.less)
    if len(envcut[0]) == 0:
        return mhplus
    return mhplus[envcut[0][0] :]


def wt(postmerger, sam_p, getfreqs=False):
    sam_f = 1 / sam_p
    scales = np.arange(scale_min, scale_max, dscale)

    # CWT on the gwf using the Morlet wavelet
    coefs, freqs = pywt.cwt(postmerger, scales, "morl", sampling_period=sam_p)

    # Normalising the coefficient matrix using the Frobenius norm
    Z = (np.abs(coefs)) / (np.linalg.norm(coefs))
    Z = Z[::5, ::45][:, :400]
    if getfreqs:
        return Z, freqs
    return Z


def numshift_pad_width(Z, l=FIN_WIDTH, shift=0):
    cwidth = Z.shape[1]
    leftpad = int((l - cwidth) / 2)
    rightpad = int((l - cwidth) / 2)
    fudgepad = rightpad + int(l - (leftpad + rightpad + cwidth))
    assert (shift <= leftpad) and (
        shift <= rightpad
    ), "Shift must be less than or equal to the padding on either side"
    padb = np.zeros((l, leftpad + shift))
    pada = np.zeros((l, fudgepad - shift))
    return np.concatenate((padb, Z, pada), axis=1)


def pad_width(Z, l=FIN_WIDTH, percent_shift=0):
    maxshift = int((l - Z.shape[1]) / 2)
    shift = percent_shift * maxshift if percent_shift != 0 else 0
    return numshift_pad_width(Z, shift=shift)


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
        randsamp=1.0,
    ):
        self.sel_attr = sel_attr
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
                current_h5_filepath = p.Path(self.sims[sim_key].run[run_key].data.path)
                current_h5_file = h5.File(current_h5_filepath / "data.h5", "r")
                current_rh_waveforms = [
                    i for i in current_h5_file.keys() if i == "rh_22"
                ]
                for selected_wf in current_rh_waveforms:
                    extraction_radii = list(
                        current_h5_file[selected_wf].keys()
                    )  # type: ignore
                    # for extraction_radius in extraction_radii:
                    #     self.eoss.append(run.md.data["id_eos"])
                    #     inserter = (sim_key, run_key, selected_wf,
                    #                 extraction_radius)
                    self.eoss.append(run.md.data["id_eos"])
                    inserter_base = [
                        sim_key,
                        run_key,
                        selected_wf,
                        extraction_radii[-1],
                    ]
                    # shiftpercents = range(0, 101, 5)  # creates 101 shifts
                    shiftpercents = [0]
                    for i in shiftpercents:
                        inserter = inserter_base.copy()
                        inserter.append(i)
                        self.pspace.append(inserter)
        # ----
        self.pspace = tuple(self.pspace)
        self.ueoss, self.ueosscounts = np.unique(
            np.array(self.eoss), return_counts=True
        )
        setueoss = list(self.ueoss)
        self.eosmap = {i: setueoss.index(i) for i in self.ueoss}
        self.numeoss = len(self.eoss)
        self.pspace = random.sample(self.pspace, int(randsamp * len(self.pspace)))
        print(self.pspace)

    def __len__(self):
        return len(self.pspace)

    def __getitem__(self, index: int):
        return self.preprocess(self.retrieve(self.pspace[index]))

    def retrieve(self, psl):
        data = self.sims[psl[0]].run[psl[1]]
        h5path = p.Path(data.data.path) / "data.h5"
        metadata = {i: data.md.data[i] for i in self.sel_attr}
        data = h5.File(h5path, "r")[psl[2]][psl[3]]  # type: ignore
        pm_time = data[:, -1]  # type: ignore
        data = cut_at_lowest_envelope(data[:, 1], data[:, 2])  # type: ignore
        sam_p = (pm_time[-1] - pm_time[0]) / len(pm_time)  # type: ignore
        return data, metadata, sam_p, psl[4]

    def preprocess(self, data):
        return self.preprocess_ts(data[0], data[2], data[3]).to(
            self.device
        ), self.preprocess_params(data[1]).to(self.device)

    def preprocess_ts(self, ts, sam_p, percent_shift):
        lts = len(ts)
        print(lts)
        if lts == 0:
            rts = np.zeros(90)
            rts[45] = ts[0]
            ts = rts
        elif lts < p2:
            ts = np.concatenate(
                (np.zeros(math.floor(p2 - lts)), ts, np.zeros(math.ceil(p2 - lts))),
                axis=0,
            )
            print("contingency")
        win = planck_window(math.floor(math.log(len(ts) / 2) * 6), len(ts))

        # return torch.tensor(pad_width(wt(ts, sam_p), percent_shift)).to(torch.float64)
        return torch.tensor(wt(ts, sam_p))

    def preprocess_params(self, params):
        outlist = [
            self.eosmap[params["id_eos"]],
            float(params["id_mass_starA"]),
            float(params["id_mass_starB"]),
            # float(get(power(signal))),
        ]
        return torch.tensor(outlist).to(torch.float32)


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_dataset(dsdevice=device, sync=False, rsamp=1.0):
    return CoRe_DB_Dataset(sync=sync, device=dsdevice, randsamp=rsamp)


dataset = get_dataset()


def to_pth_file(fname="processed_spectrograms.pth"):
    btime = time.time()
    len_dataset = len(dataset)
    print("finished with Initialization")
    print("pspace: ", len(dataset.pspace))
    spectrograms = []
    params = []
    for i, (spectrogram, param) in enumerate(dataset):  # type: ignore
        print(i, len_dataset, spectrogram.shape, (time.time() - btime) / 60)
        spectrograms.append(spectrogram)
        params.append(param)
    spectrogram_tensor = torch.stack(spectrograms)
    params_tensor = torch.stack(params)
    torch.save([spectrogram_tensor, params_tensor], fname)


def load_pth_file(
    fname="processed_spectrograms.pth",
    train_dl_batch_size=16,
    test_dl_batch_size=128 * 2,
):
    spectrograms, parameters = torch.load("processed_spectrograms.pth")
    tensor_dataset = TensorDataset(spectrograms, parameters)
    train_dl = DataLoader(tensor_dataset, batch_size=train_dl_batch_size)
    test_dl = DataLoader(tensor_dataset, batch_size=test_dl_batch_size)
    return train_dl, test_dl


def load_raw_from_pth_file(fname="processed_spectrograms.pth"):
    spectrograms, parameters = torch.load(fname)
    tensor_dataset = TensorDataset(spectrograms, parameters)
    return spectrograms, parameters


if __name__ == "__main__":
    to_pth_file(fname="nodup_with_shift_processed_spectrograms.pth")
    a, b = load_pth_file()
    print(len(a), len(b))
    # type: ignore
