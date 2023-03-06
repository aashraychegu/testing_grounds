from CoRe_Dataloader_ECSG import load_raw_from_pth_file
from torch.utils.data import DataLoader, Dataset
import torch


class CoRe_Dataset_RNoise(Dataset):
    def __init__(self, length=None, mult_length=None, sgpath="./processed_spectrograms.pth", snr=0):

        self.spectrograms, self.params = load_raw_from_pth_file(sgpath)
        self.spectrograms *= (1/torch.max(self.spectrograms))
        self.raw_length = len(self.spectrograms)
        if mult_length is not None:
            self.length = mult_length*self.raw_length
        elif length is not None:
            self.length = length
        else:
            self.length = self.raw_length
        assert self.length >= len(
            self.spectrograms), "seriously, the length of the spectrograms should be larger than the length of the spectrogram"
        self.snr = snr
        self.maxval = torch.max(self.spectrograms).item()

    def __getitem__(self, index):
        wrap_index = index % self.raw_length
        spectrogram = self.spectrograms[wrap_index]
        params = self.params[wrap_index]
        randnoise = self.create_random_noise(torch.max(spectrogram).item())
        return spectrogram + randnoise.to("cuda:0"), params

    def create_random_noise(self, maxval):
        return torch.normal(mean=0.0, std=maxval/4, size = (400, 400)) * self.snr
    def __len__(self):
        return self.length


if __name__ == "__main__":
    dataset=CoRe_Dataset_RNoise(snr = 0)
    print(dataset.maxval)
    import matplotlib.pyplot as plt
    plt.pcolormesh(dataset[1][0].cpu().numpy())
    plt.show()
