from CoRe_Dataloader_ECSG import load_raw_from_pth_file
from torch.utils.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt
import math


def calculate_std(a: torch.Tensor, snr: float):
    return ((torch.mean(a**2)/snr)**0.5).item()


def mknoise(a: torch.Tensor, snr: float):
    std = calculate_std(a, snr)
    return torch.normal(0, std, a.shape), std


def pltT(a):
    plt.pcolormesh(a.cpu().numpy())
    plt.colorbar()
    plt.show()


class CoRe_Dataset_RNoise(Dataset):
    def __init__(self, length=None, mult_length=None, sgpath="./processed_spectrograms.pth", snrs=None):

        self.spectrograms, self.params = load_raw_from_pth_file(sgpath)
        self.raw_length = len(self.spectrograms)
        if snrs is None:
            snrs = [i/100 for i in range(1, 401, 5)]
            snrs.append(0)
        self.snrlength = len(snrs)
        index_map = []
        for i in range(self.raw_length):
            for j in snrs:
                index_map.append((i, j))
        self.index_map = tuple(index_map)
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
        return spectrogram + noise.to("cuda:0"), params

    def __len__(self):
        return self.length


if __name__ == "__main__":
    dataset = CoRe_Dataset_RNoise()
    print(len(dataset))
