import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt

Series_Length = 6284

g_input_size = 3
g_hidden_size = 1500
g_output_size = Series_Length

d_input_size = Series_Length
d_hidden_size = 1500
d_output_size = 1

d_minibatch_size = 15
g_minibatch_size = 10
num_epochs = 500

d_learning_rate = 3e-3
g_learning_rate = 8e-3


def get_real_sampler(start, end, step, func=np.sin):
    x = np.linspace(start * step, end * step, Series_Length)
    y = func(x)
    # print(f"{start, start + Series_Length,len(x),len(y)}")
    # assert len(x) == len(y)
    # assert len(x) == Series_Length
    # assert len(y) == Series_Length
    return x, y, len(y)


def rand_data():
    start = np.random.uniform(0, 2000)
    end = start + 6284
    a = get_real_sampler(start, end, 1 / 1000)
    return a[0], a[1], a[2], start, end


def get_noise_sampler():
    return lambda m, n: torch.rand(
        m, n
    ).requires_grad_()  # Uniform-dist data into generator, _NOT_ Gaussian


noise_data = get_noise_sampler()

# if True:
#     actual_data = rand_data()
#     print(actual_data[2], np.argmax(actual_data[1]))
#     plt.plot(actual_data[0], actual_data[1])
#     plt.show()


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            torch.nn.SELU(),
            nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            torch.nn.SELU(),
            nn.Linear(hidden_size, hidden_size),
            torch.nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            torch.nn.SELU(),
            nn.Linear(hidden_size, hidden_size),
            torch.nn.LeakyReLU(),
        )
        self.map3 = nn.Linear(hidden_size, output_size)
        self.xfer = torch.nn.SELU()

    def forward(self, x):
        x = self.xfer(self.map1(x))
        x = self.xfer(self.map2(x))
        return self.xfer(self.map3(x))


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return torch.sigmoid(self.seq(x))


G = Generator(
    input_size=g_input_size, hidden_size=g_hidden_size, output_size=Series_Length
)
D = Discriminator(
    input_size=Series_Length, hidden_size=d_hidden_size, output_size=d_output_size
)

criterion = nn.BCELoss()
d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate)
g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate)


def train_D_on_actual():
    t = torch.empty(15, 1)
    for i in range(d_minibatch_size):
        real_data = rand_data()
        real_decision = D(torch.tensor(real_data[1], dtype=torch.float))
        t[i] = real_decision
    # print(t, torch.ones(d_minibatch_size, 1), sep="\n")
    real_error = criterion(t, torch.ones(d_minibatch_size, 1))  # ones = true
    real_error.backward()


def train_D_on_generated():
    noise = noise_data(d_minibatch_size, g_input_size)
    fake_data = G(noise)
    fake_decision = D(fake_data)
    fake_error = criterion(
        fake_decision, torch.zeros(d_minibatch_size, 1)
    )  # zeros = fake
    fake_error.backward()


def train_G():
    noise = noise_data(g_minibatch_size, g_input_size)
    fake_data = G(noise)
    fake_decision = D(fake_data)
    error = criterion(fake_decision, torch.ones(g_minibatch_size, 1))
    error.backward()
    return error.item(), fake_data


losses = []
for epoch in range(num_epochs):
    D.zero_grad()

    train_D_on_actual()
    train_D_on_generated()
    d_optimizer.step()

    G.zero_grad()
    loss, generated = train_G()
    g_optimizer.step()

    losses.append(loss)
    print("Epoch %6d. Loss %5.3f" % (epoch + 1, loss))
    # plt.plot(G(torch.tensor([0, 2 * np.pi, 1 / 1000])).detach().numpy())
print("Training complete")


def draw(data):
    plt.figure()
    d = data.tolist() if isinstance(data, torch.Tensor) else data
    plt.plot(d)
    plt.show()


# d = torch.empty(generated.size(0), 53)
# for i in range(0, d.size(0)):
#     d[i] = torch.histc(generated[i], min=0, max=5, bins=53)
# draw(d.t())

rdat = rand_data()

a = G(torch.tensor([rdat[3], rdat[4], 1 / 1000]))
x = rdat[0]
y = rdat[1]
plt.plot(x, a.detach().numpy())
plt.plot(x, y)
plt.show()
