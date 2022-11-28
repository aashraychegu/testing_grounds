import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


d_learning_rate = 3e-1
g_learning_rate = 8e-1
num_epochs = 500


def random_data(return_params=False):
    start = int(np.random.uniform(0, 180))
    # print(start)
    y = np.linspace(0, 330, 12)
    # print(y)
    y += start
    # (y)
    if return_params:
        return y[0], y[-1], np.sin(np.radians(y))
    return np.sin(np.radians(y))


random_data_generation_works = random_data()
# plt.scatter(list(range(0, 12)), random_data_generation_works, marker="o")
# plt.show()


def get_random_noise():
    start = int(np.random.uniform(0, 90))
    end = start + 330
    return torch.tensor([start, end], dtype=torch.float)


# print(get_random_noise())


class Generator(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
    ):
        super(Generator, self).__init__()
        self.map1 = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.Tanh(),
            nn.Linear(1000, 10000),
            nn.Tanh(),
            nn.Linear(10000, 50000),
            nn.Tanh(),
            nn.Linear(50000, output_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.map1(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
            nn.GELU(),
        )

    def forward(self, x):
        return torch.sigmoid(self.seq(x))


G = Generator(
    input_size=2,
    hidden_size=128,
    output_size=12,
)
D = Discriminator(input_size=12, hidden_size=128 * 3, output_size=1)

criterion = nn.BCEWithLogitsLoss()
d_optimizer = optim.ASGD(D.parameters(), lr=d_learning_rate)
g_optimizer = optim.ASGD(G.parameters(), lr=g_learning_rate)

##
#!training ⏬⏬
##
losses = []
for epoch in range(num_epochs):
    D.zero_grad()
    #! train Discriminator on actual data
    real_data = random_data()
    t = D(torch.tensor(real_data, dtype=torch.float))
    real_error = criterion(t, torch.tensor([1.0], dtype=torch.float))  # ones = true
    real_error.backward()
    # print(f"{real_data = } | (What the discriminator gave) {t = } | {real_error = }")
    #! train discriminator on fake data
    noise = get_random_noise()
    fake_data = G(noise)
    fake_decision = D(fake_data)
    fake_error = criterion(
        fake_decision, torch.tensor([0.0], dtype=torch.float)
    )  # zeros = fake
    fake_error.backward()
    # print(f"{noise = } | {fake_data = } | {fake_decision = } | {fake_error = }")
    #! run the discriminator optimizer
    d_optimizer.step()
    #! train the generator
    G.zero_grad()
    noise = get_random_noise()
    fake_data = G(noise)
    fake_decision = D(fake_data)
    error = criterion(fake_decision, torch.tensor([1.0], dtype=torch.float))
    error.backward()
    # print(f"{noise = } | {fake_data = } | {fake_decision = } | {error = }")
    #! optimize generator
    g_optimizer.step()
    #! print metrics
    loss = error.item()
    losses.append(loss)
    print("Epoch %6d. Loss %5.3f \n\n" % (epoch + 1, loss))

print("Training complete")


# d = torch.empty(generated.size(0), 53)
# for i in range(0, d.size(0)):
#     d[i] = torch.histc(generated[i], min=0, max=5, bins=53)
# draw(d.t())

start, end, ydat = random_data(True)
a = G(torch.tensor([start, end], dtype=torch.float))
x = np.linspace(0, 330, 12) + start
plt.plot(list(range(0, 12)), a.detach().numpy())
plt.scatter(list(range(0, 12)), ydat)
plt.show()
print("actual: ", ydat)
print("generated: ", a.detach().numpy())
