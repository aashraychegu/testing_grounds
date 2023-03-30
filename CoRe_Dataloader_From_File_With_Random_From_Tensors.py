from CoRe_Dataloader_ECSG import load_raw_from_pth_file
from torch.utils.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
import time
import random
from typing import *


def calculate_std(a: torch.Tensor, snr: float):
    return ((torch.mean(a**2) / snr) ** 0.5).item()


def mknoise(a: torch.Tensor, snr: float):
    std = calculate_std(a, snr)
    return torch.normal(0, std, a.shape), std


def pltT(a):
    plt.pcolormesh(a.cpu().numpy())
    plt.colorbar()
    plt.show()


class CoRe_Dataset_RNoise(Dataset):
    def __init__(self, sgs, params, device="cuda:0", snrs=None, input_index_map=[]):
        self.device = device
        self.spectrograms, self.params = sgs, params
        self.raw_length = len(self.spectrograms)

        if snrs is None:
            snrs = [i / 200 for i in range(1, 1001, 5)]
            snrs.append(0)
        self.snrlength = len(snrs)
        self.index_map = []
        if input_index_map == []:
            index_map = []
            for i in range(self.raw_length):
                for j in snrs:
                    index_map.append((i, j))
            self.index_map = index_map
        else:
            self.index_map = input_index_map
        self.length = len(self.index_map)
        self.snrs = snrs

    def __getitem__(self, index):
        sgindex, snr = self.index_map[index]
        spectrogram = self.spectrograms[sgindex].to(torch.float64)
        params = self.params[sgindex]
        params = torch.cat(
            (params, torch.tensor([snr]).to(device=self.device)))
        std = (torch.mean(spectrogram**2) / snr) ** 0.5
        if snr == 0:
            std = 0
        noise = torch.normal(0, std, spectrogram.shape)
        return spectrogram.to(self.device) + noise.to(self.device), params

    def __len__(self):
        return self.length


def to_pth_file(
    dataset,
    fname="processed_spectrograms.pth",
):
    btime = time.time()
    len_dataset = len(dataset)
    print("finished with Initialization")
    spectrograms = []
    params = []
    for i, (spectrogram, param) in enumerate(dataset):  # type: ignore
        print(i, len_dataset, spectrogram.shape, (time.time() - btime) / 60)
        spectrograms.append(spectrogram)
        params.append(param)
    spectrogram_tensor = torch.stack(spectrograms)
    params_tensor = torch.stack(params)
    torch.save([spectrogram_tensor, params_tensor], fname)


def get_new_test_train_validation_datasets(test_split=0.1, valid_split=0.1):
    sgrams, params = load_raw_from_pth_file()
    original_ds = CoRe_Dataset_RNoise(sgrams, params)
    length = len(original_ds)
    # print(length)
    original = original_ds.index_map
    del original_ds
    p = test_split + valid_split
    testval_set = set(random.sample(original, math.ceil(length * p)))
    train_set = set(original) - testval_set

    test_set = set(random.sample(list(testval_set),
                   math.ceil(length * test_split)))
    valid_set = set(testval_set) - test_set

    train_ds = CoRe_Dataset_RNoise(
        sgrams, params, input_index_map=list(train_set))
    test_ds = CoRe_Dataset_RNoise(
        sgrams, params, input_index_map=list(test_set))
    valid_ds = CoRe_Dataset_RNoise(
        sgrams, params, input_index_map=list(valid_set))

    return train_ds, test_ds, valid_ds


def get_new_ttv_dataloaders(test_split=0.1, valid_split=0.1):
    train_ds, test_ds, valid_ds = get_new_test_train_validation_datasets(
        test_split=test_split, valid_split=valid_split
    )
    return (
        DataLoader(train_ds, batch_size=16, shuffle=True),
        DataLoader(valid_ds, batch_size=40, shuffle=True),
        DataLoader(test_ds, batch_size=40, shuffle=True),
    )


if __name__ == "__main__":
    # train_dl, valid_dl, test_dl = get_new_ttv_dataloaders(0,0)
    train_dl, valid_dl, test_dl = get_new_test_train_validation_datasets(0, 0)

    print(len(train_dl), len(test_dl), len(valid_dl), "\n")
    print(train_dl[5][1])
