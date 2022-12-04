import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint

dfbackend = mp.get_backend()


def rand_data(expxmod=1.0, sinxmod=1.0, len_in_pis=3, return_values=False):
    start = int(np.random.uniform(0, 200))
    end = start + int(np.pi * 1000 * len_in_pis)
    x = np.arange(start, end) / 1000
    y = np.sin(np.exp(x * expxmod) * sinxmod)
    if return_values:
        return y, start, end
    return y


def return_sg(
    expxmod=1,
    sinxmod=1,
    length=3.0,
    switch="agg",
):
    data = rand_data(expxmod, len_in_pis=length)
    plt.switch_backend(switch) if switch != None else print("using a normal backend")
    powerSpectrum, _, _, _ = plt.specgram(
        data, NFFT=128 * 2, noverlap=111, pad_to=64 * 2 - 1
    )

    return powerSpectrum


def plot_sg(data):
    import matplotlib.pyplot as plt

    plt.switch_backend(dfbackend)
    fig = plt.pcolormesh(data, hatch="/", cmap="plasma")
    plt.xlim(0, data.shape[0])
    return fig


class specGds(Dataset):
    def __init__(self, num_samples: int, scale_factor: float, start: int, numpis=3):
        self.slist = np.arange(1, num_samples + 1, 1, dtype=np.float64)
        self.slist *= scale_factor
        self.slist = self.slist + start
        self.numpis = numpis
        print(self.__str__())

    def __len__(self):
        return len(self.slist)

    def __getitem__(self, index):
        # np.random.seed(index)
        return torch.tensor(
            return_sg(
                expxmod=self.slist[index],
                sinxmod=randint(0, 1000) / 20,
                length=self.numpis,
            ),
            dtype=torch.float,
        ).view((1, 64, 64))

    def plotitem(self, index):
        plot_sg(self.__getitem__(index)[0])
        plt.show()

    def __str__(self) -> str:
        return f"{self.slist.shape = } | {self.numpis = } | {self.__len__() = }"
