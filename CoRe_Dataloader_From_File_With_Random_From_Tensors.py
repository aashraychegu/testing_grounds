from CoRe_Dataloader_ECSG import load_raw_from_pth_file
from torch.utils.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
import time
import random


def calculate_std(a: torch.Tensor, snr: float):
    return ((torch.mean(a**2)/snr)**0.5).item()


def mknoise(a: torch.Tensor, snr: float):
    std = calculate_std(a, snr)
    return torch.normal(0, std, a.shape), std


def pltT(a):
    plt.pcolormesh(a.cpu().numpy())
    plt.colorbar()
    plt.show()


def get_new_test_train_datasets(p=0.3):
    raw_sgram_ds, raw_param_ds = load_raw_from_pth_file()
    xtrain, xtest, ytrain, ytest = train_test_split(
        raw_sgram_ds.cpu().numpy(), raw_param_ds.cpu().numpy(), test_size=p)
    # print(xtrain.shape, xtest.shape)
    train_dataset = CoRe_Dataset_RNoise(
        torch.tensor(xtrain), torch.tensor(ytrain))
    test_dataset = CoRe_Dataset_RNoise(
        torch.tensor(xtest), torch.tensor(ytest))
    return train_dataset, test_dataset


class CoRe_Dataset_RNoise(Dataset):
    def __init__(self, sgs, params, device="cuda:0", snrs=None, input_index_map=[]):
        self.device = device
        self.spectrograms, self.params = sgs, params
        self.raw_length = len(self.spectrograms)
        if snrs is None:
            snrs = [i/100 for i in range(1, 501, 3)]
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
        std = ((torch.mean(spectrogram**2)/snr)**0.5)
        if snr == 0:
            std = 0
        noise = torch.normal(0, std, spectrogram.shape)
        return spectrogram.to(self.device) + noise.to(self.device), params

    def __len__(self):
        return self.length


def to_pth_file(dataset, fname="processed_spectrograms.pth",):
    btime = time.time()
    len_dataset = len(dataset)
    print("finished with Initialization")
    spectrograms = []
    params = []
    for i, (spectrogram, param) in enumerate(dataset):  # type: ignore
        print(i, len_dataset, spectrogram.shape, (time.time()-btime)/60)
        spectrograms.append(spectrogram)
        params.append(param)
    spectrogram_tensor = torch.stack(spectrograms)
    params_tensor = torch.stack(params)
    torch.save([spectrogram_tensor, params_tensor], fname)


def get_new_test_train_dataloaders(p=.1):
    sgrams, params = load_raw_from_pth_file()
    train_ds = CoRe_Dataset_RNoise(sgrams, params)
    length = len(train_ds)
    original = train_ds.index_map
    sampled = set(random.sample(original, math.floor(length*p)))
    substracted = set(original)-sampled
    test_ds = CoRe_Dataset_RNoise(sgrams, params, input_index_map=sampled)
    train_ds = CoRe_Dataset_RNoise(sgrams, params, input_index_map=substracted)
    print(len(sampled), len(substracted))
    return DataLoader(train_ds, batch_size=20, shuffle=True), DataLoader(test_ds, batch_size=128, shuffle=True)


if __name__ == "__main__":
    train_dl, test_dl = get_new_test_train_dataloaders()
    print(len(train_dl), len(test_dl))
