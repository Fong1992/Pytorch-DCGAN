import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(channels_noise, features_g * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 4, features_g * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 8, features_g, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    show_num = 16
    batch_size = 256
    image_size = 64
    image_channel = 3
    noise_channel = 150

    num_workers = 10
    lr = 0.0002
    num_epochs = 200
    real_label = 1.
    fake_label = 0.

    transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    data_dir = 'D:/Data/DataSet/Gan_Cat'

    dataset = datasets.ImageFolder(os.path.join(data_dir), transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    netD = Discriminator(image_channel, image_size).to(device)
    netG = Generator(noise_channel, image_channel, image_size).to(device)
    netD.apply(weights_init)
    netG.apply(weights_init)
    # from torchsummary import summary
    # summary(netD, (image_channel, image_size, image_size))
    # summary(netG, (noise, image_channel, image_channel))

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    netG.train()
    netD.train()
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, noise_channel, 1, 1, device=device)

    step = 0
    writer_real = SummaryWriter(f"tfboard/test_real")
    writer_fake = SummaryWriter(f"tfboard/test_fake")

    for epoch in range(num_epochs):
        for batch_idx, (data, _) in enumerate(dataloader):

            data = data.to(device)
            b_size = data.size(0)

            #### Train Discriminator ####
            optimizerD.zero_grad()
            label = torch.full((b_size,), real_label, device=device)
            output = netD(data).view(-1)
            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, noise_channel, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            lossD_fake = criterion(output, label)
            lossD_fake.backward()
            optimizerD.step()
            lossD = lossD_real + lossD_fake

            #### Train Generator ####
            optimizerG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            lossG = criterion(output, label)
            lossG.backward()
            optimizerG.step()

            if batch_idx % 10 == 0:
                step += 1
            print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {lossD.item():.4f}, loss G: {lossG.item():.4f} D(x): {D_x:.4f}")

            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                img_grid_real = torchvision.utils.make_grid(data[:show_num], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:show_num], normalize=True)
                writer_real.add_image("Real Images", img_grid_real, global_step=step)
                writer_fake.add_image("Fake Images", img_grid_fake, global_step=step)
