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
        f = np.sin(freqfactor * np.exp(preexpfac * x))
        return f

    def wf_to_sg(self, ra):
        _, _, Sxx = sg(ra, 10e4, nfft=128 * 4, nperseg=38, noverlap=15)
        return Sxx[:256:4, :256:4]

    def gen_SG(self, start=None, end=None, length=None, preexpfac=1, freqfactor=1):
        start = self.start if start is None else start
        end = self.end if end is None else end
        length = end * 1000 if length is None else length
        return self.wf_to_sg(self.genExS(start, end, length, preexpfac, freqfactor))


class specGds(Dataset):
    def __init__(
        self,
        device,
        start,
        end,
        scaleFactor=1 / 1000,
        genStart=0,
        genEnd=pi3,
        genLength=None,
    ):
        self.device = device
        self.pspace = np.linspace(start, end, int((end - start) / scaleFactor))
        self.gen = ExS(genStart, genEnd, genLength)
        print(self.__str__())

    def __len__(self):
        return len(self.pspace)

    def __getitem__(self, index):
        a = tensor(
            self.gen.gen_SG(freqfactor=self.pspace[index]), device=self.device
        ).float()
        return a.view((1, 64, 64))

    def plotitem(self, index):
        plt.pcolormesh(self.__getitem__(index)[0])
        plt.show()

    def __str__(self) -> str:
        return f"{self.pspace.shape = } | {self.__len__() = }"


if __name__ == "__main__":
    ds = specGds(
        "cuda:0",
        6,
        7,
    )
    s = DataLoader(ds, batch_size=1000)
    a = next(iter(s))
    A = a[999][0]
    AA = A.clone()
    AA = AA.view(A.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.view(A.size(0), A.size(1))

    np.set_printoptions(threshold=np.inf)

    print(torch.round(AA, decimals=3).cpu().numpy())
    # print(torch.round(, decimals=3).cpu().numpy())
    plt.pcolormesh(AA.cpu())
    plt.show()
