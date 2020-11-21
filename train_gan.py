import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm

from models.discriminator import Discriminator
from models.generator import Generator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def update_discriminator(batch, discriminator, generator, optimizer, params):
    batch_size = batch.size(0)

    optimizer.zero_grad()
    
    # predictions on generating distribution
    labels_real = torch.ones(batch_size, 1, device=batch.device)
    preds_real = discriminator(batch)
    loss_real = F.binary_cross_entropy(preds_real, labels_real)

    # predictions on fake distribution
    labels_fake = torch.zeros(batch_size, 1, device=batch.device)
    latent = torch.randn(batch_size, params["dim_latent"], device=batch.device)
    batch_fake = generator(latent)
    
    preds_fake = discriminator(batch_fake.detach())
    loss_fake = F.binary_cross_entropy(preds_fake, labels_fake)
    
    loss = loss_real + loss_fake
    loss.backward()
    optimizer.step()
    
    return loss

def update_generator(discriminator, generator, optimizer, params, device):
    optimizer.zero_grad()

    labels_real = torch.ones(params["num_fake_samples"], 1, device=device)
    latent = torch.randn(params["num_fake_samples"], params["dim_latent"], device=device)
    batch_fake = generator(latent)
    
    preds_fake = discriminator(batch_fake)
    loss = F.binary_cross_entropy(preds_fake, labels_real)
    loss.backward()
    optimizer.step()
    
    return loss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, required=True, help="path to MNIST dataset folder")
    parser.add_argument("--params", type=str, default="./examples/params.json",
                        help="path to hyperparameters")
    parser.add_argument("--logdir", type=str, default="./tensorboard",
                        help="directory storing tensorboard logs")
    parser.add_argument("--modeldir", type=str, default="./trained_models",
                        help="directory storing all during training saved models")
    parser.add_argument("-o", type=str, default="gan", help="model's name")
    parser.add_argument("--device", type=int, default=0, help="gpu device to use")

    return parser.parse_args()

def main(args):
    with open(args.params, "r") as f:
        params = json.load(f)
    
    generator = Generator(params["dim_latent"])
    discriminator = Discriminator()

    if args.device is not None:
        generator = generator.cuda(args.device)
        discriminator = discriminator.cuda(args.device)

    # dataloading
    train_dataset = datasets.MNIST(root=args.datadir, transform=transforms.ToTensor(), train=True)
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"],
                                num_workers=4, shuffle=True)

    # optimizer
    betas = (params["beta_1"], params["beta_2"])
    optimizer_G = optim.Adam(generator.parameters(), lr=params["learning_rate"], betas=betas)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=params["learning_rate"], betas=betas)

    if not os.path.exists(args.modeldir): os.mkdir(args.modeldir)
    if not os.path.exists(args.logdir): os.mkdir(args.logdir)
    writer = SummaryWriter(args.logdir)

    steps_per_epoch = len(train_loader)

    # main training loop
    for n in range(params["num_epochs"]):
        loader = iter(train_loader)

        for i in tqdm.trange(steps_per_epoch):
            batch, _ = next(loader)
            if args.device is not None: batch = batch.cuda(args.device)
            
            loss_D = update_discriminator(batch, discriminator, generator, optimizer_D, params)
            loss_G = update_generator(discriminator, generator, optimizer_G,
                                        params, args.device)

            writer.add_scalar("loss_discriminator/train", loss_D, i + n * steps_per_epoch)
            writer.add_scalar("loss_generator/train", loss_G, i + n * steps_per_epoch)
        
        torch.save(generator.state_dict(), args.o + ".generator." + str(n) + ".tmp")
        torch.save(discriminator.state_dict(), args.o + ".discriminator." + str(n) + ".tmp")

        # eval
        latent = torch.randn(8, params["dim_latent"]).cuda()
        imgs_fake = generator(latent)
        writer.add_images("generated fake images", imgs_fake, n)
            
    writer.close()

    torch.save(generator.state_dict(), args.o + ".generator.pt")
    torch.save(discriminator.state_dict(), args.o + ".discriminator.pt")


if __name__ == '__main__':
    args = get_args()
    main(args)