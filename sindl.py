import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import tensor, float as tfloat
import matplotlib.pyplot as plt
from scipy.signal import spectrogram as sg

pi3 = round(2 * np.pi, 3)


class ExS:
    def __init__(
        self,
        start=0,
        end=pi3,
        length=None,
    ) -> None:
        self.start = start
        self.end = end
        self.length = end * 1000 if length is None else length

    def genExS(self, start, end, numsamples, preexpfac=2, freqfactor=1):
        x = np.linspace(start, end, int(numsamples))
        f = np.sin(freqfactor * x)
        return f

    def gen_SG(self, start=None, end=None, length=None, preexpfac=1, freqfactor=1):
        start = self.start if start is None else start
        end = self.end if end is None else end
        length = end * 1000 if length is None else length
        return self.genExS(start, end, length, preexpfac, freqfactor)


class specGds(Dataset):
    def __init__(
        self,
        start,
        end,
        num_samples=None,
        genStart=0,
        genEnd=pi3,
        genLength=None,
        device=None,
    ):
        if device == None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        num_samples = int((end - start) * 1000) if num_samples is None else num_samples
        self.pspace = np.linspace(start, end, num_samples)
        self.gen = ExS(genStart, genEnd, genLength)
        print(self.__str__())

    def __len__(self):
        return len(self.pspace)

    def __getitem__(self, index):
        return tensor(
            self.gen.gen_SG(freqfactor=self.pspace[index]), device=self.device
        ).float()

    def plotitem(self, index):
        plt.pcolormesh(self.__getitem__(index)[0])
        plt.show()

    def __str__(self) -> str:
        return f"{self.pspace.shape = } | {self.__len__() = }"


dataset = specGds(1, 6, num_samples=3200)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print(dataset[1].shape)