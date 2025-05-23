import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from fontTools.subset import retain_empty_scripts
from torch.nn import Sequential
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, img_dim=784, hidden_dim=128):
        super().__init__()
        self.disc = nn.Sequential(
            # Hidden layers
            self.get_discriminator_block(input_dim=img_dim, output_dim=hidden_dim * 4),
            self.get_discriminator_block(input_dim=hidden_dim * 4, output_dim=hidden_dim * 2),
            self.get_discriminator_block(input_dim=hidden_dim * 2, output_dim=hidden_dim),
            # Output layers
            nn.Linear(in_features=hidden_dim, out_features=1),
            # nn.Sigmoid()
        )
    def forward(self, x):
        return self.disc(x)
    def get_discriminator_block(self, input_dim, output_dim) -> Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

class Generator(nn.Module):
    def __init__(self, z_dim=64, img_dim=784, hidden_dim=128):
        super().__init__()
        self.gen = nn.Sequential(
            # Hidden layers
            self.get_generator_block(input_dim=z_dim, output_dim=hidden_dim),
            self.get_generator_block(input_dim=hidden_dim, output_dim=hidden_dim * 2),
            self.get_generator_block(input_dim=hidden_dim * 2, output_dim=hidden_dim * 4),
            self.get_generator_block(input_dim=hidden_dim * 4, output_dim=hidden_dim * 8),
            # Output layer
            nn.Linear(in_features=hidden_dim * 8, out_features=img_dim),
            nn.Sigmoid() # Tanh()
        )
    def forward(self, noise):
        return self.gen(noise)
    def  get_generator_block(self, input_dim, output_dim) -> Sequential:
        return nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim),
            nn.BatchNorm1d(num_features=output_dim),
            nn.ReLU(inplace=True),
        )

# Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 3e-4
z_dim = 64
img_dim = 784
batch_size = 32
num_epochs = 50

disc = Discriminator(img_dim=img_dim).to(device)
gen = Generator(z_dim=z_dim, img_dim=img_dim).to(device)
fixed_noise = torch.randn(size=(batch_size, z_dim)).to(device)
# transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

dataset = datasets.MNIST(root='../../datasets/', transform=transforms.ToTensor(), download=False)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(params=disc.parameters(), lr=lr)
opt_gen = optim.Adam(params=gen.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()
writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
writer_real = SummaryWriter(f'runs/GAN_MNIST/real')
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        cur_batch_size = real.shape[0]
        real = real.view(cur_batch_size, -1).to(device)

        ### Train the Discriminator:                                max log(D(real)) + log(1 - D(G(z)))
        noise = torch.randn(size=(cur_batch_size, z_dim)).to(device)
        fake = gen(noise)
        disc_real = disc(real)
        disc.zero_grad()
        loss_D_real = criterion(disc_real, torch.ones_like(disc_real)) #log(D(real))
        disc_fake = disc(fake.detach())
        loss_1_DGz = criterion(disc_fake, torch.zeros_like(disc_fake)) #log(1 - D(G(z)))
        lossD = (loss_D_real + loss_1_DGz) / 2
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train the Generator:                                    min log(1 - D(G(z)))  <-->  saturating gradient = no training
        ###                                                                               <-->  max log(D(G(z)) to avoid that
        # disc_fake2 = disc(fake)
        gen.zero_grad()
        lossG = criterion(disc_fake, torch.ones_like(disc_fake))
        lossG.backward(retain_graph=True)
        opt_gen.step()

        # Tensorboard
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Loss D: {lossD:.3f}, Loss G: {lossG:.3f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)
                writer_fake.add_image("Mnist Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Mnist real Images", img_grid_real, global_step=step)
                step += 1
