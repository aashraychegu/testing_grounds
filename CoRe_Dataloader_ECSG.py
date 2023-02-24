from torch.utils.data import DataLoader, Dataset
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

scale_min = 1
scale_max = 201
dscale = 0.1
FIN_WIDTH= 400

def cut_at_lowest_envelope(hplus, hcross):
    # Cutting inspiral off
    oenv = np.sqrt(hplus**2 + hcross**2)
    cut_point = np.argmax(hplus)
    mhplus = hplus[cut_point:]
    env = oenv[cut_point:]
    envcut = argrelextrema(env,np.less)
    if len(envcut[0])==0:
        return mhplus
    return mhplus[envcut[0][0]:]

def wt(postmerger,sam_p):
    sam_f = 1/sam_p    
    scales = np.arange(scale_min, scale_max, dscale)

    #CWT on the gwf using the Morlet wavelet
    coefs, freqs = pywt.cwt(postmerger, scales, 'morl', sampling_period = sam_p)

    #Normalising the coefficient matrix using the Frobenius norm
    Z = (np.abs(coefs))/(np.linalg.norm(coefs))
    Z = Z[::5,::45][:,:400]
    return Z

def pad_width(Z,l = FIN_WIDTH):
    cwidth = Z.shape[1]
    padb = np.zeros((l,int((l-cwidth)/2)))
    pada = np.zeros((l,int((l-cwidth)/2)))
    Z = np.concatenate((padb,Z,pada),axis = 1)
    cwidth = Z.shape[1]
    fpad = np.zeros((l,int((l-cwidth))))
    return np.concatenate((Z,fpad),axis = 1)


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
        randsamp = 1,
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
                current_h5_filepath = p.Path(
                    self.sims[sim_key].run[run_key].data.path)
                current_h5_file = h5.File(current_h5_filepath / "data.h5", "r")
                current_rh_waveforms = [i for i in current_h5_file.keys() if i == "rh_22"]
                for selected_wf in current_rh_waveforms:
                    extraction_radii = list(current_h5_file[selected_wf].keys()) # type: ignore                    
                    for extraction_radius in extraction_radii:
                        self.eoss.append(run.md.data["id_eos"])
                        inserter = (sim_key, run_key, selected_wf, extraction_radius)
                        self.pspace.append(inserter)
        #----                
        self.pspace = tuple(self.pspace)
        self.ueoss,self.ueosscounts = np.unique(np.array(self.eoss),return_counts=True)
        setueoss = list(self.ueoss)
        self.eosmap = {i: setueoss.index(i) for i in self.ueoss}
        self.numeoss = len(self.eoss)
        self.pspace = random.sample(self.pspace,int(randsamp * len(self.pspace)))

    def __len__(self):
        return len(self.pspace)

    def __getitem__(self, index: int):
        return self.preprocess(self.retrieve(self.pspace[index]))

    def retrieve(self, psl):
        data = self.sims[psl[0]].run[psl[1]]
        h5path = p.Path(data.data.path) / "data.h5"
        metadata = {i: data.md.data[i] for i in self.sel_attr}
        data = h5.File(h5path, "r")[psl[2]][psl[3]] # type: ignore
        pm_time = data[:,-1] # type: ignore
        data = cut_at_lowest_envelope(data[:, 1], data[:, 2]) # type: ignore
        sam_p = (pm_time[-1] - pm_time[0])/len(pm_time) # type: ignore
        return data, metadata, sam_p,

    def preprocess(self, data):
        return self.preprocess_ts(data[0], data[2]).to(self.device), self.preprocess_params(
            data[1]
        ).to(self.device)

    def preprocess_ts(self, ts,sam_p):
        z = pad_width(wt(ts,sam_p))
        return torch.tensor(z).to(torch.float16)

    def preprocess_params(self, params):
        outlist = [self.eosmap[params["id_eos"]],
        float(params["id_mass_starA"]),
        float(params["id_mass_starB"])]
        return torch.tensor(outlist).to(torch.float32)
    

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = CoRe_DB_Dataset(sync=False, device=device)

def get_dataset(dsdevice = device,sync = False,rsamp = 1.):
    return CoRe_DB_Dataset(sync=sync, device=dsdevice,randsamp=rsamp)
    

if __name__ == "__main__":
    print(len(dataset))
    lens = []
    for i in dataset:
        lens.append(i[1][0].cpu().item())
    print(set(lens))
    print(np.unique(np.array(lens)))