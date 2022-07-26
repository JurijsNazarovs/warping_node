import torch.nn as nn
import math
import os
import pickle
import numpy as np
import torch
from sklearn import metrics

from torch.utils.data import DataLoader, TensorDataset
import cv2
#from skimage.transform import rotate
import torchvision.transforms as transforms
import random
import torch.nn.functional as F

from PIL import Image


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def combine_batch(x, fac=1, imgw=1216):
    #fac is how many images to comnbine

    if torch.is_tensor(x):
        x = torch.split(x, fac, 0)  #split on batches of images to combine
        x = torch.stack(
            x, axis=0)  #now, each x[i] have to be combined with proper shape
        x = torch.split(x, imgw // x.shape[-1], 1)
        x = torch.stack(x, axis=0)
        x = x.permute(1, 3, 0, 4, 2, 5)
    else:
        x = np.split(x, fac, 0)
        x = np.stack(x, axis=0)

        dims = tuple(range(len(x.shape)))
        x = np.transpose(x, (0, ) + dims[2:-1] + (1, dims[-1]))

    x = x.reshape(x.shape[:2] + (-1, imgw))
    return x


def normalize(x, q=None, method='scale', xmin=None, xmax=None):

    if method == 'mean':
        # mu/sigma standartization. If q is provided,
        # compute mu, sigma on those only
        if q is not None:
            max_per_x = x.amax(dim=[2, 3])
            if torch.is_tensor(max_per_x):
                ind = max_per_x <= torch.quantile(max_per_x, q)
            else:
                ind = max_per_x <= np.quantile(max_per_x, q)
            ind = ind.reshape(-1, x.shape[1])  #(-1)
            x_ = x[ind]
        else:
            x_ = x
        mu = x_.mean()  #dim=0)
        std = x_.std()  #dim=0)
        print("mu: %f, std: %f" % (mu, std))
        print("Before norm: ", x.min(), x.max())
        x = (x - mu) / std
        print("After norm: ", x.min(), x.max())
        mu = x.mean()  #dim=0)
        std = x.std()  #dim=0)
        print("After norm mu: ", mu.min(), mu.max())
        print("After norm std: ", std.min(), std.max())

    elif method == 'scale':
        # final result is from 0 to 1
        if xmin is None or xmax is None:
            xmax = torch.amax(x, (1, 2, 3), keepdim=True)  #.detach()
            xmin = torch.amin(x, (1, 2, 3), keepdim=True)  #.detach()
        x = (x - xmin) / (xmax - xmin)
    elif method == 'sigmoid':
        # final result is from 0 to 1
        if torch.is_tensor(x):
            x = torch.nn.functional.sigmoid(x)
    elif method == 'clip' and q is not None:
        # Clipping based on quantile values
        clipval = torch.quantile(x.reshape(x.shape[0], -1), q, dim=1)
        for i in range(len(clipval)):
            x[i] = torch.clip(x[i], max=clipval[i])
    else:
        raise ValueError("Unknown normalization method:", method)

    return x


def save_model(args, model, ckpt_path, epoch=0, best_loss=np.infty):
    generator = model.gen
    discriminator = model.discr

    torch.save(
        {
            'args': args,
            'state_dict_gen': generator.state_dict(),
            'state_dict_discr': discriminator.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
        }, ckpt_path)


def load_model(ckpt_path, model, device, layers=['gen', 'discr']):
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    else:
        print("Loading model from %s" % ckpt_path)
    # Load checkpoint.
    checkpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = checkpt['args']  # Not used?
    epoch_st = checkpt['epoch']
    best_loss = checkpt['best_loss']
    #if 'best_loss' in checkpt.keys() else np.infty

    for layer in layers:
        state_dict = checkpt['state_dict_' + layer]
        _model = getattr(model, layer)
        model_dict = _model.state_dict()
        # 1. filter out unnecessary keys
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # if layer == "gen":
        #     state_dict['jac_weight'] = torch.ones(1)
        #     state_dict['outgrid_weight'] = torch.ones(1)

        # 2. overwrite entries in the existing state dict
        model_dict.update(state_dict)
        # 3. load the new state dict
        _model.load_state_dict(state_dict)
        _model.to(device)
    #del checkpt
    return epoch_st, best_loss


def make_plots(img,
               mask_hat,
               mask=None,
               img_fake=None,
               plots_path='',
               mask_cochran=None,
               mask_quantile=None,
               base_img=None):
    os.makedirs(os.path.dirname(plots_path), exist_ok=True)
    #mask_hat = torch.bitwise_and(
    #    mask_hat == 1, img > torch.quantile(img, 0)).to(torch.int64)  #0.8

    rfc.plot_rf(img, plot_path="%s_rf.png" % plots_path)
    rfc.plot_rf(mask_hat,
                plot_path="%s_mask_hat.png" % plots_path,
                is_mask=True)
    rfc.plot_rf(img,
                mask=mask_hat,
                plot_path="%s_rf_mask_hat.png" % plots_path)

    if mask is not None:
        rfc.plot_rf(-mask, plot_path="%s_mask.png" % plots_path, is_mask=True)
        overlap = torch.bitwise_and(mask_hat == 1, mask == 1)
        diff = (mask_hat - mask).to(torch.float)
        diff[overlap] += 0.3
        rfc.plot_rf(diff,
                    plot_path="%s_mask_hat_vs_true.png" % plots_path,
                    is_mask=True)

    if img_fake is not None:
        # Fake images: make sense for source input
        rfc.plot_rf(img_fake, plot_path="%s_fake.png" % plots_path)

    if mask_cochran is not None:
        rfc.plot_rf(mask_cochran,
                    plot_path="%s_mask_cochran.png" % plots_path,
                    is_mask=True)
        if mask is not None:
            rfc.plot_rf(mask_cochran - mask,
                        plot_path="%s_mask_cochran_vs_true.png" % plots_path,
                        is_mask=True)

    if mask_quantile is not None:
        rfc.plot_rf(mask_quantile,
                    plot_path="%s_mask_q.png" % plots_path,
                    is_mask=True)
        if mask is not None:
            overlap = torch.bitwise_and(mask_quantile == 1, mask == 1)
            diff = (mask_quantile - mask).to(torch.float)
            diff[overlap] += 0.3
            rfc.plot_rf(diff,
                        plot_path="%s_mask_q_vs_true.png" % plots_path,
                        is_mask=True)

    if base_img is not None:
        # plot mask above base_img
        print(base_img.shape)
        if base_img.shape[0] == 3:
            #base_img = base_img.permute(1, 2, 0).type(torch.int64)
            base_img = base_img.data.cpu().numpy()
            base_img = np.transpose(base_img, (1, 2, 0))
            if base_img.max() > 2:
                base_img = base_img / 255

            rfc.plot_rf(
                base_img,  #.astype(np.uint8),
                plot_path="%s_base.png" % plots_path)
            #base_img = cv2.cvtColor(base_img,
            #                        cv2.COLOR_BGR2GRAY)  #.astype(np.uint8)
            print(base_img.shape)

        rfc.plot_rf(base_img,
                    mask=mask_hat,
                    plot_path="%s_base_h.png" % plots_path,
                    base_cmap='Greys')

        if mask_quantile is not None:
            rfc.plot_rf(base_img,
                        mask=mask_quantile,
                        plot_path="%s_base_q.png" % plots_path,
                        base_cmap='Greys')


# ------------------------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------------------------


def get_data(path, device="cpu", n_files_read=100, grey_scale=False):
    """
    Function to load random fields from directory
    """
    if os.path.isdir(path):
        data = []
        i = 0
        for f in os.listdir(path):
            i += 1
            if i > n_files_read:
                break

            if not os.path.isfile("%s/%s" % (path, f)):
                continue
            try:
                batch = pickle.load(open("%s/%s" % (path, f), "rb"))
            except:
                try:
                    img = Image.open("%s/%s" % (path, f))

                    # Crop the center of the image
                    width, height = img.size  # Get dimensions
                    new_width, new_height = 300, 300
                    left = (width - new_width) / 2
                    top = (height - new_height) / 2
                    right = (width + new_width) / 2
                    bottom = (height + new_height) / 2
                    img = img.crop((left, top, right, bottom))

                    img = img.resize((64, 64))
                    if grey_scale:
                        img = img.convert('L')
                        batch = np.asarray(img)[None, None]
                    else:
                        batch = np.asarray(img).transpose(2, 0, 1)[None]
                except:
                    raise ValueError("Unknown format of file: %s" % f)
            data.append(batch)

        data = np.concatenate(data)
        print("Read data from the %s of the following shape %s" %
              (path, data.shape))
    else:
        data = pickle.load(open(path, "rb"))
    #return torch.tensor(data[:10 * 95], dtype=torch.float32).to(device)
    return torch.tensor(data, dtype=torch.float32).to(device)


def create_dataloader(path_data,
                      path_mask=None,
                      path_base=None,
                      batch_size=1,
                      is_normalize=False,
                      shuffle=True,
                      normalize_method='scale',
                      mode='full',
                      n_files_read=100):
    # mode:
    #  - 'full' - full data is used,
    #  - 'train'- 0.7 left used,
    #  - 'test' - 0.3 right used
    device = 'cpu'  # to make sure that GPU is not occupied with all data
    data = get_data(path_data, device, n_files_read)

    if is_normalize:
        ind = range(len(data))

        train_frac = 0.7
        if mode == 'train':
            ind = range(int(len(ind) * train_frac))
        elif mode == 'test':
            ind = range(int(len(ind) * train_frac), len(ind))
        else:
            print("The full dataset is used")

        data = data[ind]  #filtering data
        data = normalize(data, q=1, method=normalize_method)  #default is scale

    seed = random.randint(0, 2**24)
    set_seed(seed)  # to shuffle base in a right way

    def _init_fn(worker_id):
        torch.initial_seed()

    batch_size = min(data.shape[0], batch_size)

    loader = DataLoader(data,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        worker_init_fn=_init_fn,
                        num_workers=0,
                        drop_last=True)

    #print(data.min())
    #print(data.max())
    print(len(data))
    return loader, len(data)


def inf_generator(iterable):
    """
    Allows training with DataLoaders in a single infinite loop:
    for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


import torch.autograd as autograd


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA=10):
    BATCH_SIZE = real_data.shape[0]

    alpha = torch.rand(BATCH_SIZE, 1, 1, 1).to(real_data)
    alpha = alpha.repeat([1, 1] + list(real_data.shape[-2:]))

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(
            disc_interpolates.size()).to(real_data),  #.cuda(gpu)
        #if use_cuda else torch.ones(disc_interpolates.size()),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    gradient_penalty = ((1. - torch.sqrt(1e-8 + torch.sum(
        gradients.view(gradients.size(0), -1)**2, dim=1)))**2).mean() * LAMBDA
    if torch.isnan(gradient_penalty):
        import pdb
        pdb.set_trace()
        print("DEBUG! Disc loss is nan")
    return gradient_penalty


def init_network_weights(m):
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)


def get_jacdet2d_filter(displacement, grid=None, backward=False):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*2*h*w
    '''
    # # Apply Sobell 3x3 filter
    # # Dx
    # a = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).to(displacement)
    # a = a.view((1, 1, 3, 3))

    # # Dy
    # b = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(displacement)
    # b = b.view((1, 1, 3, 3))

    # # Apply Robert 2x2 filter
    # # Dx
    # a = torch.Tensor([[1, 0], [0, -1]]).to(displacement)
    # a = a.view((1, 1, 2, 2))

    # # Dy
    # b = torch.Tensor([[0, 1], [-1, 0]]).to(displacement)
    # b = b.view((1, 1, 2, 2))

    # Apply average derivative 2x2 filter
    # Dx
    a = 1 / 2 * torch.Tensor([[-1, 1], [-1, 1]]).to(displacement)
    a = a.view((1, 1, 2, 2))

    # Dy
    b = 1 / 2 * torch.Tensor([[1, 1], [-1, -1]]).to(displacement)
    b = b.view((1, 1, 2, 2))

    # Take derivative
    Dx_x = F.conv2d(displacement[:, 0:1], a)
    Dx_y = F.conv2d(displacement[:, 0:1], b)

    Dy_x = F.conv2d(displacement[:, 1:], a)
    Dy_y = F.conv2d(displacement[:, 1:], b)

    # Normal grid
    if backward:
        D1 = (1 - Dx_x) * (1 - Dy_y)
    else:
        D1 = (1 + Dx_x) * (1 + Dy_y)
    D2 = Dx_y * Dy_x
    jacdet = D1 - D2

    return jacdet


def get_jacdet2d(displacement, grid=None, backward=False):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*2*h*w
    '''
    Dx_x = (displacement[:, 0, :-1, 1:] - displacement[:, 0, :-1, :-1])
    Dx_y = (displacement[:, 0, 1:, :-1] - displacement[:, 0, :-1, :-1])
    Dy_x = (displacement[:, 1, :-1, 1:] - displacement[:, 1, :-1, :-1])
    Dy_y = (displacement[:, 1, 1:, :-1] - displacement[:, 1, :-1, :-1])

    # Dy_x = (displacement[:, 0, :-1, 1:] - displacement[:, 0, :-1, :-1])
    # Dy_y = (displacement[:, 0, 1:, :-1] - displacement[:, 0, :-1, :-1])
    # Dx_x = (displacement[:, 1, :-1, 1:] - displacement[:, 1, :-1, :-1])
    # Dx_y = (displacement[:, 1, 1:, :-1] - displacement[:, 1, :-1, :-1])

    # Normal grid
    if backward:
        D1 = (1 - Dx_x) * (1 - Dy_y)
    else:
        D1 = (1 + Dx_x) * (1 + Dy_y)
    #D1 = (Dx_x) * (Dy_y)
    D2 = Dx_y * Dy_x
    jacdet = D1 - D2

    # # tanh grid
    # grid_x = grid[:, 0, :-1, :-1]
    # grid_y = grid[:, 1, :-1, :-1]
    # coef = 1 - torch.tanh(torch.atanh(grid) + displacement)**2
    # coef_x = coef[:, 0, :-1, :-1]
    # coef_y = coef[:, 1, :-1, :-1]
    # D1 = (1 / (1 - grid_x**2) + Dx_x) * (1 / (1 - grid_y**2) + Dy_y)
    # D2 = Dx_y * Dy_x
    # jacdet = coef_x * coef_y * (D1 - D2)

    return jacdet


def get_jacdet3d(displacement):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    raise NotImplementedError("Need to fix")

    D_y = (displacement[:, 1:, :-1, :-1, :] -
           displacement[:, :-1, :-1, :-1, :])

    D_x = (displacement[:, :-1, 1:, :-1, :] -
           displacement[:, :-1, :-1, :-1, :])

    D_z = (displacement[:, :-1, :-1, 1:, :] -
           displacement[:, :-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) *
                              (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])

    D2 = (D_x[..., 1]) * (D_y[..., 0] *
                          (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])

    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] -
                          (D_y[..., 1] + 1) * D_z[..., 0])

    return D1 - D2 + D3


def jacdet_loss(vf, grid=None, backward=False):
    '''
    Penalizing locations where Jacobian has negative determinants
    Add to final loss
    '''
    jacdet = get_jacdet2d(vf, grid, backward)
    ans = 1 / 2 * (torch.abs(jacdet) - jacdet).mean(axis=[1, 2]).sum()
    return ans


def outgrid_loss(vf, grid, backward=False, size=32):  #32-1):
    '''
    Penalizing locations where Jacobian has negative determinants
    Add to final loss
    '''
    if backward:
        pos = grid - vf - (size - 1)
        neg = grid - vf
    else:
        pos = grid + vf - (size - 1)
        neg = grid + vf

    # penalize > size
    ans_p = 1 / 2 * (torch.abs(pos) + pos).mean(axis=[1, 2]).sum()
    # penalize < 0
    ans_n = 1 / 2 * (torch.abs(neg) - neg).mean(axis=[1, 2]).sum()
    ans = ans_n + ans_p

    return ans


def get_running(old_value, new_value, i):
    value = (old_value * i + new_value) / (i + 1)
    return value


def get_beta(beta_type="original",
             n_batches=1,
             batch_idx=1,
             reverse=False,
             weight=1):
    """
    Function returns beta for VI inference
    """

    if beta_type == "blundell":
        # https://arxiv.org/abs/1505.05424
        beta = 2**(n_batches - (batch_idx)) / (2**n_batches - 1)
    elif beta_type == "graves":
        # https://papers.nips.cc/paper/2011/file/7eb3c8be3d411e8ebfab08eba5f49632-Paper.pdf
        # eq (18)
        beta = 1 / n_batches
    elif beta_type == "cycle":
        beta = frange_cycle_linear(n_batches, n_cycle=1,
                                   ratio=0.5)[batch_idx] * 0.001
    elif beta_type == "default":
        beta = weight
    else:
        beta = 0

    if reverse:
        beta = 1 - beta
    return beta


def frange_cycle_linear(n_epoch, start=0., stop=1., n_cycle=4, ratio=0.5):
    # From here:  github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
    # n_epochs can also be n_batches
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_epoch):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def check_grads(model, model_name):
    grads = []
    for p in model.parameters():
        if not p.grad is None:
            grads.append(float(p.grad.mean()))

    grads = np.array(grads)
    if grads.any() and grads.mean() > 100:
        print("********\n"
              "WARNING! gradients mean is over 100", model_name, "\n********")
    if grads.any() and grads.max() > 100:
        print("********\n"
              "WARNING! gradients max is over 100", model_name, "\n********")


def aug_trans(data, mask=None):
    # if torch.is_tensor(data):
    #     data = data.numpy()

    transform = transforms.Compose([
        transforms.RandomAffine(degrees=180, translate=None),  #(0.1, 0.1)),
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomHorizontalFlip(p=0.2)
    ])
    if mask is not None:
        state = torch.get_rng_state()
        data = transform(data)
        torch.set_rng_state(state)
        mask = transform(mask)
        return (data, mask)
    else:
        return transform(data)
