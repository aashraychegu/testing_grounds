import numpy as np
import matplotlib.pyplot as plt
import os

os.system("cls")


def gen2Dline(m=1, end=64):
    array = np.zeros((end, end))

    x = np.linspace(0, end - 1, end)
    # print(x, x.shape)
    y = m * x

    for xc, yc in zip(x, y):
        if yc < end:
            array[int(yc)][int(xc)] = 1
    return array


np.set_printoptions(threshold=np.inf)
array = gen2Dline(m=2, end=32)


print(array)


plt.pcolormesh(array)
plt.show()
