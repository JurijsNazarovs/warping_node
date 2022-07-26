"""
File describes model, where the main component is Warping Neural ODE
"""
from models.warp_node import STODE

import torch
import torch.nn as nn
import utils


def weights_init(m):
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)


class WarpNODEModel():
    def __init__(self, args, device='cpu'):
        if args.disc_type == "conv":
            from models.discriminator import DiscriminatorFConv as Discriminator
        else:
            from models.discriminator import Discriminator as Discriminator

        self.discr = Discriminator(args).to(device)
        self.gen = STODE(args, device).to(device)

        self.device = device

        self.discr.apply(weights_init)
        self.gen.apply(weights_init)

        if args.gan_type == 'gan':
            self.gan_crit = nn.BCELoss  # no ()
        elif args.gan_type == 'lsgan':
            self.gan_crit = nn.MSELoss  #lsGan
        else:
            self.gan_crit = nn.MSELoss  #None

        self.class_crit = nn.BCEWithLogitsLoss  # BSELoss but then need sigmoid

    def get_n_parameters(self):
        total = 0
        for model in [self.gen, self.discr]:
            total += sum(p.numel() for p in model.parameters()
                         if p.requires_grad)

        print("Total number of parameters: %d" % total)
