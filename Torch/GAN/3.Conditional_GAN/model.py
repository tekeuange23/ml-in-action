# Implementation https://arxiv.org/pdf/1511.06434v1.pdf
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, n_classes):
        """
            n_classes: the total number of classes in the dataset, an integer scalar.   e.g : 3 == label.shape[0]
        """
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # No Normalization in the first layer. Input: N * channels_img * 64 * 64
            nn.Conv2d(in_channels=channels_img+1, out_channels=features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            # Normalization
            self._block(in_channels=features_d, out_channels=features_d*2, kernel_size=4, stride=2, padding=1),
            self._block(in_channels=features_d*2, out_channels=features_d*4, kernel_size=4, stride=2, padding=1),
            self._block(in_channels=features_d*4, out_channels=features_d*8, kernel_size=4, stride=2, padding=1),
            # output layer:
            nn.Conv2d(in_channels=features_d*8, out_channels=1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )
        self.img_size = features_d
        self.channels_img = channels_img
        self.embed = nn.Embedding(num_embeddings=n_classes, embedding_dim=features_d*features_d)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, labels):
        """
            labels: tensor of labels from the dataloader, size (?).                         e.g: torch.Tensor([[0, 2, 1]]).long()
        """
        embedding = self.embed(labels=labels).view(labels.shape[0], self.channels_img, self.img_size, self.img_size)   # (N,   C, H, W)
        x = torch.cat(tensors=[x, embedding], dim=1)                                                                   # (N, C+1, H, W)     +1 class info
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, n_classes):
        """
            n_classes: the total number of classes in the dataset, an integer scalar.   e.g : 3 == label.shape[0]
        """
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: z_dim
            self._block(in_channels=z_dim+1, out_channels=features_g*16, kernel_size=4, stride=1, padding=0),
            self._block(in_channels=features_g*16, out_channels=features_g*8, kernel_size=4, stride=2, padding=1),
            self._block(in_channels=features_g*8, out_channels=features_g*4, kernel_size=4, stride=2, padding=1),
            self._block(in_channels=features_g*4, out_channels=features_g*2, kernel_size=4, stride=2, padding=1),
            # Output layer
            nn.ConvTranspose2d(in_channels=features_g*2, out_channels=channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.channels_img = channels_img
        self.img_size = features_g
        self.embed = nn.Embedding(num_embeddings=n_classes, embedding_dim=z_dim)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, z, labels):
        """
            z: latent vector (N, z_dim, 1, 1)
            labels: tensor of labels from the dataloader, size (?).                         e.g: torch.Tensor([[0, 2, 1]]).long()
        """
        embedding = self.embed(labels=labels).unsqueeze(dim=2).unsqueeze(dim=3)     # (N, z_dim, 1, 1)
        z = torch.cat(tensors=[z, embedding], dim=1)                                                                   # (N, C+1, 1, 1)     +1 class info
        return self.gen(z)

def init_weights(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(layer.weight.data, 0.0, 0.02)

if __name__ == '__main__':
    N, in_channels, H, W = 16, 3, 64, 64
    z_dim = 100
    num_classes = 10
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(channels_img=in_channels, features_d=N, n_classes=num_classes)
    init_weights(disc)
    assert disc(x, 3).shape == (N, 1, 1, 1)

    gen = Generator(z_dim=z_dim, channels_img=in_channels, features_g=N, n_classes=num_classes)
    noises = torch.randn((N, z_dim, 1, 1))
    init_weights(gen)
    assert gen(noises, 3).shape == (N, in_channels, H, W)

    print('Success!')

