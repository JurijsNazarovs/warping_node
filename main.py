import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import itertools

import os
import sys
import pickle
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter

from config import get_arguments
from data_config import get_datapath
import utils
from models.warp_node_model import WarpNODEModel


class Logger(object):
    def __init__(self, path='output.log'):
        self.terminal = sys.stdout
        self.log = open(path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def train(args,
          source_loader,
          target_loader,
          model,
          optims,
          n_batches=1,
          epoch=0,
          ckpt_path='./ckpt',
          best_loss=None,
          tb_writer=None,
          train_gen=True,
          train_disc=True,
          train_reverse=True):
    device = model.device
    #-------------------------------------------------------------------
    # Define models and optimizers
    #-------------------------------------------------------------------

    generator, discriminator = model.gen, model.discr

    generator.train()
    discriminator.train()
    optim_G, optim_D = optims

    #-------------------------------------------------------------------
    # Define criteria loss for model
    #-------------------------------------------------------------------
    gan_crit = model.gan_crit()

    #-------------------------------------------------------------------
    running_gen_loss = 0.0
    running_rec_loss = 0.0
    running_disc_loss = 0.0
    running_disc_acc = 0.0
    running_disc_acc_truth = 0.0
    running_disc_acc_fake = 0.0
    running_jacdet_forw = 0.0
    running_jacdet_back = 0.0
    running_outgrid_forw = 0.0
    running_outgrid_back = 0.0

    for i in range(n_batches):
        start_time_batch = time.time()
        args.rec_loss_weight = utils.get_beta(args.rec_weight_method,
                                              n_batches,
                                              i,
                                              reverse=False,
                                              weight=args.rec_loss_weight)
        #print("rec_loss_weight: ", args.rec_loss_weight)

        imgs_s = source_loader.__next__()

        if args.augment:
            imgs_s = utils.aug_trans(imgs_s)
        imgs_s = imgs_s.to(model.device)

        imgs_t = target_loader.__next__()
        imgs_t = imgs_t.to(device)

        N = min(imgs_t.shape[0], imgs_s.shape[0])  #batch_size
        imgs_s, imgs_t = imgs_s[:N], imgs_t[:N]

        # --------------
        # Train Generator: F -> Z
        # --------------
        truth_label = torch.ones((N, 1)).to(device)
        false_label = torch.zeros((N, 1)).to(device)

        imgs_fake_t = generator(imgs_s, y=imgs_t)
        if args.gan_type == 'wgan':
            # https://arxiv.org/pdf/1704.00028.pdf page 4
            gen1_loss = -discriminator(imgs_fake_t).mean()
        elif args.gan_type in ['gan', 'lsgan']:
            gen1_loss = gan_crit(discriminator(imgs_fake_t), truth_label)
        else:
            # MSELoss/L1Loss
            gen1_loss = nn.MSELoss(reduction='sum')(imgs_fake_t, imgs_t)

        gen_loss = gen1_loss / N * args.gen_loss_weight
        running_gen_loss = utils.get_running(running_gen_loss, gen_loss.item(),
                                             i)

        # Add jacobian determinant constrain
        if args.jac_loss_weight_forw != 0:
            gen_loss += generator.stode_grad.jacob / N * args.jac_loss_weight_forw
        running_jacdet_forw = utils.get_running(
            running_jacdet_forw,
            generator.stode_grad.jacob.item() / N, i)
        #print('jacob_loss forward:', generator.stode_grad.jacob / N)

        # Add outside of the grid constrain (to keep vf inside the grid)
        if args.outgrid_loss_weight_forw != 0:
            gen_loss += generator.stode_grad.outgrid_loss / N * args.outgrid_loss_weight_forw
        running_outgrid_forw = utils.get_running(
            running_outgrid_forw,
            generator.stode_grad.outgrid_loss.item() / N, i)
        #print('outgrid_loss forward:', generator.stode_grad.outgrid_loss / N)

        if i == 0:
            print("\nn steps: %d" % model.gen.de_solver.nfe)

        model.gen.de_solver.nfe = 0  #reset counter of ode steps
        if train_gen:
            # -----------------------
            # Reverse Generator Start
            def get_rec_loss():
                imgs_fake_s = generator(
                    imgs_fake_t,  #.detach(),
                    y=None,
                    backward=True,
                    init_img=imgs_s)
                rec_loss_ = nn.MSELoss(reduction='sum')(imgs_fake_s, imgs_s)
                #recov_loss_ = nn.L1Loss(reduction='sum')(imgs_fake_s, imgs_s)
                rec_loss = rec_loss_ / N * args.rec_loss_weight
                # Below cannot use the same name, because it is local variable
                # so I add _ at the end: running_rec_loss_
                running_rec_loss_ = utils.get_running(running_rec_loss,
                                                      rec_loss_.item() / N, i)
                return (rec_loss, running_rec_loss_)

            if args.rec_loss_weight != 0 and train_reverse:
                recov_loss, running_rec_loss = get_rec_loss()

                # Add jacobian determinant constrain
                if args.jac_loss_weight_back != 0:
                    recov_loss += generator.stode_grad.jacob / N * args.jac_loss_weight_back
                running_jacdet_back = utils.get_running(
                    running_jacdet_back,
                    generator.stode_grad.jacob.item() / N, i)
                #print('jacob_loss backward:', generator.stode_grad.jacob / N)

                # Add outside of the grid constrain (to keep vf inside the grid)
                if args.outgrid_loss_weight_back != 0:
                    recov_loss += generator.stode_grad.outgrid_loss / N *\
                        args.outgrid_loss_weight_back
                running_outgrid_back = utils.get_running(
                    running_outgrid_back,
                    generator.stode_grad.outgrid_loss.item() / N, i)
                # print('outgrid_loss backward:',
                #       generator.stode_grad.outgrid_loss / N)

                gen_loss += recov_loss
            else:
                # Run without gradient to keep results in summary
                with torch.no_grad():
                    recov_loss, running_rec_loss = get_rec_loss()

            # Reverse Generator End
            # -----------------------

            utils.check_grads(generator, 'Generator')
            optim_G.zero_grad()
            gen_loss.backward()

            if args.gen_clip_grad != 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(),
                                               args.gen_clip_grad)
            optim_G.step()

        # --------------
        # Train Discriminator,
        # by fixing \theta_G, \theta_C, and optimize for \theta_D
        # --------------
        for crit_iter in range(args.crit_iter):

            truth_label = torch.ones((N, 1)).to(device)
            false_label = torch.zeros((N, 1)).to(device)
            ## ---------- Tricks to train GAN ----------
            # Add noise labels eacho N iterations (by swapping lables)
            if i % 10 == 0:
                noise_p = 0.15
                ind_add_noise = np.random.choice(
                    len(truth_label), int(noise_p * len(truth_label)))
                truth_label[ind_add_noise] = false_label[ind_add_noise]

                ind_add_noise = np.random.choice(
                    len(truth_label), int(noise_p * len(truth_label)))
                false_label[ind_add_noise] = truth_label[ind_add_noise]
            ## ---------- Tricks to train GAN END ----------

            if args.gan_type == 'wgan':
                gradient_penalty = utils.calc_gradient_penalty(
                    discriminator,
                    imgs_t,
                    imgs_fake_t.detach(),
                    LAMBDA=args.gplambda)

                ld = -discriminator(imgs_t).mean() +\
                    discriminator(imgs_fake_t.detach()).mean()+ gradient_penalty
            elif args.gan_type in ['gan', 'lsgan']:
                ld = gan_crit(discriminator(imgs_t), truth_label) +\
                    gan_crit(discriminator(imgs_fake_t.detach()), false_label)
            else:
                ld = torch.zeros(1).to(imgs_t)

            disc_loss = ld * args.disc_loss_weight
            if torch.isnan(disc_loss):
                import pdb
                pdb.set_trace()
                print("DEBUG! Disc loss is nan")

            if train_disc and args.gan_type != 'None':
                # We skip this part in case it is L2 loss and not Gan
                utils.check_grads(discriminator, 'Discriminator')
                optim_D.zero_grad()
                disc_loss.backward()
                if True:  #disc_loss.item() / N > 0.0003:  #0.009
                    if args.disc_clip_grad != 0:
                        torch.nn.utils.clip_grad_norm_(
                            discriminator.parameters(), args.disc_clip_grad)
                    optim_D.step()
                else:
                    print('No discriminator update')
                    optim_D.zero_grad()
                running_disc_loss = utils.get_running(running_disc_loss,
                                                      disc_loss.item() / N, i)

        end_time_batch = time.time()
        #print("Batch_time: %f" % (end_time_batch - start_time_batch))

        # --------------
        # Summaries
        # --------------
        if args.gan_type == 'wgan':
            func = lambda x: torch.round(torch.clip(x, 0, 1))
        elif args.gan_type == 'lsgan':
            func = lambda x: torch.round(torch.clip(x, 0, 1))
        else:
            func = lambda x: torch.round(x)

        disc_acc_truth = (
            (func(discriminator(imgs_t.detach()))
             == torch.round(truth_label)).sum()) / len(truth_label)

        disc_acc_fake = (
            (func(discriminator(imgs_fake_t.detach()))
             == torch.round(false_label)).sum()) / len(false_label)

        disc_acc = ((func(discriminator(imgs_t.detach())) == torch.round(truth_label)).sum() +\
            (func(discriminator(imgs_fake_t.detach())) ==\
             torch.round(false_label)).sum())/(len(truth_label) + len(false_label))
        running_disc_acc = utils.get_running(running_disc_acc, disc_acc, i)
        running_disc_acc_truth = utils.get_running(running_disc_acc_truth,
                                                   disc_acc_truth, i)
        running_disc_acc_fake = utils.get_running(running_disc_acc_fake,
                                                  disc_acc_fake, i)

        print(
            "[ Batch %04d/%04d] [ Gen Loss: %05.8f ] [ Disc Loss:  %05.8f ] [ Disc acc:  %05.4f ] [ Running Disc acc:  %05.4f ] [ Rec Loss: %05.4f]"
            % (i + 1, n_batches, running_gen_loss, running_disc_loss, disc_acc,
               running_disc_acc, running_rec_loss),
            end='\r')  #,
        #flush=True)
        # Save every batch
        utils.save_model(args, model, ckpt_path + '_batch', epoch, best_loss)

    # Save every epochs
    print()
    utils.save_model(args, model, ckpt_path, epoch, best_loss)

    if best_loss is None:
        best_loss = np.infty

    # write loss to tb
    if tb_writer is not None:
        tb_writer.add_scalar('gen_loss', running_gen_loss, epoch)
        tb_writer.add_scalar('discr_loss', running_disc_loss, epoch)
        tb_writer.add_scalar('discr_acc', running_disc_acc, epoch)
        tb_writer.add_scalar('discr_acc_truth', running_disc_acc_truth, epoch)
        tb_writer.add_scalar('discr_acc_fake', running_disc_acc_fake, epoch)
        tb_writer.add_scalar('rec_loss', running_rec_loss, epoch)
        tb_writer.add_scalar('jacdet_forw', running_jacdet_forw, epoch)
        tb_writer.add_scalar('jacdet_back', running_jacdet_back, epoch)
        tb_writer.add_scalar('outgrid_forw', running_outgrid_forw, epoch)
        tb_writer.add_scalar('outgrid_back', running_outgrid_back, epoch)

    return best_loss


def test(args,
         source_loader,
         model,
         best_loss=None,
         n_batches=1,
         epoch=0,
         ckpt_path='./ckpt',
         is_source=True,
         condition=True,
         target_loader=None):
    print("Start testing")
    torch.cuda.empty_cache()

    generator = model.gen
    generator.eval()

    # plots are saved in generator.stode_grad.plots_path
    test_loss = 0  #compute if you need it
    for batch_iter in range(n_batches):
        imgs = source_loader.__next__().to(model.device)
        # Test_imgs are needed only to plot vector field and difference
        # between results. If it is not desired, than test_imgs
        # can be None.
        if target_loader is not None:
            test_imgs = target_loader.__next__()
        else:
            test_imgs = None
        # Folder for vf_seq: warped images
        generator.stode_grad.model_name = "%d" % batch_iter
        print(batch_iter)
        # Folder for vf: last vector field
        generator.model_name = "%d" % batch_iter
        imgs_fake = generator(imgs, backward=not is_source, y=test_imgs)

    # ------------------------------------------------------------------------
    # Save best loss
    # ------------------------------------------------------------------------
    #  Below you can replace if False: with real condition,
    #  if you know how to compute best loss in your case.
    #  For this project that was not necessary, and we train until convergence.
    if False:  #best_loss is not None:
        if test_loss < best_loss and condition:
            best_loss = test_loss
            utils.save_model(args, model, ckpt_path + '_best', epoch,
                             best_loss)
            print("New best loss: %f" % best_loss)
        else:
            print("Best loss is still the same: %f" % best_loss)

        return best_loss

    print("End testing")
    return test_loss


def main():
    parser = get_arguments()
    args = parser.parse_args()
    data_path = get_datapath(args)
    args.plots_path += "%s/" % (args.data)  #also saved in the model for plots

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.makedirs(args.save, exist_ok=True)

    experimentID = args.experimentID
    if experimentID is None:
        experimentID = int(SystemRandom().random() * 100000)
    ckpt_path = os.path.join(args.save,
                             "experiment_" + str(experimentID) + '.ckpt')
    os.makedirs("results/", exist_ok=True)

    #-------------------------------------------------------------------
    # Model and optimizers
    #-------------------------------------------------------------------

    if args.test_only:
        args.ode_step_size = 0.01
        args.ode_solver = 'euler'
        args.shuffle = False
    else:
        args.plot_all = False

    if torch.cuda.is_available():
        device = torch.device('cuda')
        if args.device:
            device = "%s:%d" % (device, args.device)
    else:
        device = torch.device('cpu')
    print("Device:", device)

    model = WarpNODEModel(args, device)
    model.get_n_parameters()

    optim_G = optim.Adam(model.gen.parameters(),
                         lr=args.lr_gen,
                         weight_decay=args.l2_decay,
                         betas=(args.opt_beta_1, args.opt_beta_2))
    if args.disc_optim == 'adam':
        print("***************")
        print("Adam optimizer for discriminator")
        print("***************")
        optim_D = optim.Adam(model.discr.parameters(),
                             lr=args.lr_disc,
                             weight_decay=args.l2_decay,
                             betas=(args.opt_beta_1, args.opt_beta_2))
    else:
        print("***************")
        print("SGD optimizer for discriminator")
        print("***************")
        optim_D = optim.SGD(model.discr.parameters(), lr=args.lr_disc)

    optims = [optim_G, optim_D]
    scheduler_G = optim.lr_scheduler.StepLR(optim_G,
                                            step_size=args.decay_every,
                                            gamma=args.lr_decay)
    scheduler_D = optim.lr_scheduler.StepLR(optim_D,
                                            step_size=args.decay_every,
                                            gamma=args.lr_decay)

    #----------------------------------------------------------------------
    # Load checkpoint and evaluate the model
    #----------------------------------------------------------------------
    if args.load:
        # In case we load model to contrinue from last epoch
        # if args.best:
        #     ckpt_path_load = ckpt_path + "_best"
        # el
        if args.batch:
            ckpt_path_load = ckpt_path + "_batch"
        else:
            ckpt_path_load = ckpt_path
        epoch_st, best_loss = utils.load_model(ckpt_path_load, model, device)
        epoch_st += 1
        print("Current best loss: %.8f" % best_loss)
    else:
        epoch_st, best_loss = 1, np.infty

    #-----------------------------------------------------------------------
    # Data definition
    #-----------------------------------------------------------------------
    # data_path is exported from config.py
    utils.set_seed(0)

    source_loader, sample_size_source = utils.create_dataloader(
        path_data=data_path['source']['data'],
        batch_size=args.batch_size,
        is_normalize=True,
        shuffle=args.shuffle,
        normalize_method=args.normalize_method,
        mode='full',
        n_files_read=args.n_files_read)

    target_loader, sample_size_target = utils.create_dataloader(
        path_data=data_path['target']['data'],
        batch_size=args.batch_size,
        is_normalize=True,
        shuffle=args.shuffle,
        normalize_method=args.normalize_method,
        mode='full',
        n_files_read=args.n_files_read)

    source_loader = utils.inf_generator(source_loader)
    target_loader = utils.inf_generator(target_loader)

    # During training we would like to exploit all sampels of source,
    # which might be much larger than number of samples in target.
    # However, during final testing we use source number of samples only
    if args.test_only:
        sample_size = sample_size_source
    else:
        sample_size = max(sample_size_source, sample_size_target)

    args.n_epochs_to_viz = max(args.freq_gen_update, args.freq_disc_update)

    #-----------------------------------------------------------------------
    # Training
    log_path = "logs/" + str(experimentID) + ".log"
    os.makedirs("logs/", exist_ok=True)
    sys.stdout = Logger(log_path)
    #args.batch_size = min(args.batch_size, sample_size)
    n_batches = max(sample_size // args.batch_size, 1)

    os.makedirs("logs/tb/", exist_ok=True)
    tb_writer = SummaryWriter('logs/tb/%s' % (experimentID))

    if args.test_only:
        print("Testing results achieved at epoch %d" % epoch_st)
        with torch.no_grad():
            test(args,
                 source_loader,
                 model,
                 epoch=0,
                 n_batches=min(n_batches, args.n_test_batches),
                 best_loss=None,
                 ckpt_path=ckpt_path,
                 is_source=True,
                 target_loader=target_loader)

    else:
        for epoch in range(epoch_st):
            # In case we load model and need to update schedulers to
            # appropriate state
            scheduler_G.step()
            scheduler_D.step()

        for epoch in range(epoch_st, args.n_epochs + 1):
            print("lr_G: ", optim_G.param_groups[0]['lr'])
            print("lr_D: ", optim_D.param_groups[0]['lr'])
            print('Epoch %04d' % epoch)
            best_loss = train(args,
                              source_loader,
                              target_loader,
                              model,
                              optims,
                              n_batches=n_batches,
                              epoch=epoch,
                              ckpt_path=ckpt_path,
                              best_loss=best_loss,
                              tb_writer=tb_writer,
                              train_gen=epoch % args.freq_gen_update == 0,
                              train_disc=epoch % args.freq_disc_update == 0,
                              train_reverse=epoch % args.freq_rec_update == 0)

            scheduler_G.step()
            scheduler_D.step()

            # Do testing
            if epoch % args.n_epochs_to_viz == 0 and\
               epoch >= args.n_epochs_start_viz:
                with torch.no_grad():
                    print("-----------------")

                    print("test_loss", test_loss)
                    print("Testing run:")
                    best_loss = test(args,
                                     source_loader,
                                     model,
                                     epoch=epoch,
                                     n_batches=min(n_batches,
                                                   args.n_test_batches),
                                     best_loss=best_loss,
                                     ckpt_path=ckpt_path,
                                     is_source=True)


if __name__ == '__main__':
    main()
