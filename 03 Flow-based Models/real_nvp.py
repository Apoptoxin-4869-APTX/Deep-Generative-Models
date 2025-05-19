# main author Jakub Tomczak
# modified by Ertunc Erdil

import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# === Define Digits dataset class === #
class Digits(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(self, mode='train', transforms=None):
        digits = load_digits()
        if mode == 'train':
            self.data = digits.data[:1000].astype(np.float32)
        elif mode == 'val':
            self.data = digits.data[1000:1350].astype(np.float32)
        else:
            self.data = digits.data[1350:].astype(np.float32)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample

# === Define RealNVP class === #
class RealNVP(nn.Module):
    def __init__(self, nets, nett, num_flows, prior, D=2, dequantization=True):
        super(RealNVP, self).__init__()

        self.dequantization = dequantization
        
        self.prior = prior
        self.t = torch.nn.ModuleList([nett() for _ in range(num_flows)])
        self.s = torch.nn.ModuleList([nets() for _ in range(num_flows)])

        self.num_flows = num_flows
        self.D = D

    # Implements the coupling layer
    def coupling(self, x, index, forward=True):
        (xa, xb) = torch.chunk(x, 2, 1)
        
        s = self.s[index](xa)
        t = self.t[index](xa)
        
        if forward:
            #f
            yb = torch.exp(s) * xb + t
        else:
            #f^-1
            yb = (xb - t) * torch.exp(-s)
            
        return torch.cat((xa, yb), 1), s

    # Implements the permutation layer
    def permute(self, x):
        return x.flip(1)

    # Implements inverse transformation z=f^-1(x)
    # where x is from data distribution and z is from the base (prior) distribution
    def f_inv(self, x):
        log_det_J = x.new_zeros(x.shape[0])
        for i in range(self.num_flows):
            x = self.permute(x)
            x, s = self.coupling(x, i, forward=False)
            
            log_det_J = log_det_J - s.sum(dim=1)

        return x, log_det_J

    # Implements inverse transformation x=f(z) 
    # where x is from data distribution and z is from the base (prior) distribution
    def f(self, z):
        for i in reversed(range(self.num_flows)):
            z, _ = self.coupling(z, i, forward=True)
            z = self.permute(z)

        return z

    # Compute latent z from a given input x using inverse tranformation f^-1 and calculate the negative log-likelihood
    def forward(self, x, reduction='avg'):
        z, log_det_J = self.f_inv(x)
        if reduction == 'sum':
            return -(self.prior.log_prob(z) + log_det_J).sum()
        else:
            return -(self.prior.log_prob(z) + log_det_J).mean()

    # To generate a data sample, sample from the prior and transform it to the data distribution using forward transformation f
    def sample(self, batchSize):
        z = self.prior.sample((batchSize,))
        # breakpoint()
        # z = z[:, 0, :]
        x = self.f(z)
        return x.view(-1, self.D)

# === Function to evaluate the model === #
def evaluation(data_loader, model_path=None, model=None, epoch=None):
    """
    Evaluate the model
    Args:
        data_loader (torch.utils.data.dataloader.DataLoader): Data loader to be used for evaluation
        model_path (str): The path to the saved pretrained model
        model (RealNVP): The RealNVP object containing the model
        epoch (int): Epoch number at which the model is evaluated

    Return:
        loss (float): Average loss on the evaluation set
    """

    if model is None:
        # load best performing model
        model = torch.load(f"{model_path}/realnvp.model")

    model.eval()
    loss = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(data_loader):
        if hasattr(model, 'dequantization'):
            if model.dequantization:
                test_batch = test_batch + (1. - torch.rand(test_batch.shape))/2.
        loss_t = model.forward(test_batch, reduction='sum')
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]

    loss = loss / N

    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')

    return loss

# === Function to plot and save real samples from the dataset === #
def samples_real(path, data_loader):
    """
    Get real images from the dataloader of the original dataset and save to the specified path
    Args:
        path (str): The path to save the images
        data_loader (torch.utils.data.dataloader.DataLoader): Data loader to get the data to be saved

    Return: None
    """
    num_x = 4
    num_y = 4
    x = next(iter(data_loader)).detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(f"{path}/real_images.png", bbox_inches='tight')
    plt.close()

# === Function to generate samples from the model and save === #
def samples_generated(path, extra_name=''):
    """
    Generates samples from the model and save to the specified path
    Args:
        path (str): The path to save the images
        extra_name (str): Extra string to be added end of the saved file name

    Return: None
    """

    model_best = torch.load(f"{path}/realnvp.model")
    model_best.eval()

    num_x = 4
    num_y = 4
    x = model_best.sample(num_x * num_y)
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(f"{path}/generated_images_{extra_name}.png", bbox_inches='tight')
    plt.close()

# === Plot NLL loss curve === #
def plot_curve(path, nll_val):
    """
    Plot and save NLL loss curve
    Args:
        path (str): The path to save the curve
        nll_val (list): NLL loss values at every epochs

    Return: None
    """
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(f"{path}/nll_val_curve.png", bbox_inches='tight')
    plt.close()

# === Function to train the model === #
def training(path, max_patience, num_epochs, model, optimizer, train_loader, val_loader):
    """
    Function to train the model
    Args:
        path (str): The path to save the outputs
        max_patience (int): If training doesn't improve for longer than max_patience epochs, it is stopped
        num_epochs (int): Number of epochs to train the model
        model (ARM): The model to be trained.
        optimizer (torch.optim): Optimizer to be used during model training
        train_loader (torch.utils.data.dataloader.DataLoader): Data loader for training set
        val_loader (torch.utils.data.dataloader.DataLoader): Data loader for validation set

        nll_val (list): NLL loss values at every epochs

    Return:
        nll_val (list): The list containing NLL values during training
    """
    nll_val = []
    best_nll = 1000.
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for indx_batch, batch in enumerate(train_loader):

            if hasattr(model, 'dequantization'):
                if model.dequantization:
                    batch = batch + (1. - torch.rand(batch.shape))/2.
            loss = model.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        loss_val = evaluation(data_loader=val_loader, model=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting

        if e == 0:
            print('saved!')
            torch.save(model, f"{path}/realnvp.model")
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print('saved!')
                torch.save(model, f"{path}/realnvp.model")
                best_nll = loss_val
                patience = 0

                samples_generated(path, extra_name=f"epoch_{str(e)}")
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)

    return nll_val

def main():
    # === Parameters === #
    D = 64   # input dimension
    M = 256  # the number of neurons in scale (s) and translation (t) nets

    lr = 1e-3 # learning rate
    num_epochs = 1000 # max. number of epochs
    max_patience = 20 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped

    # The number of invertible transformations
    num_flows = 8
    batch_size = 32 # batch size

    # === Create dataset objects and dataloaders === #
    train_data = Digits(mode='train')
    val_data = Digits(mode='val')
    test_data = Digits(mode='test')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # === Define the networks === #
    # scale (s) network
    nets = lambda: nn.Sequential(nn.Linear(D // 2, M), nn.LeakyReLU(),
                                nn.Linear(M, M), nn.LeakyReLU(),
                                nn.Linear(M, D // 2), nn.Tanh())

    # translation (t) network
    nett = lambda: nn.Sequential(nn.Linear(D // 2, M), nn.LeakyReLU(),
                                nn.Linear(M, M), nn.LeakyReLU(),
                                nn.Linear(M, D // 2))

    # === Define the model === #
    # Prior (base) distribution - Unit Gaussian
    prior = torch.distributions.MultivariateNormal(torch.zeros(D), torch.eye(D))
    # Crate the RealNVP object
    model = RealNVP(nets, nett, num_flows, prior, D=D, dequantization=True)

    result_dir = './results_realnvp/'
    if not(os.path.exists(result_dir)):
        os.mkdir(result_dir)

    # Define optimizer
    optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)

    # Start training
    nll_val = training(path=result_dir, 
                       max_patience=max_patience, 
                       num_epochs=num_epochs, 
                       model=model, 
                       optimizer=optimizer,
                       train_loader=train_loader, 
                       val_loader=val_loader)
    # Evaluation
    test_loss = evaluation(data_loader=test_loader, model_path=result_dir)

    f = open(f"{result_dir}/test_loss.txt", "w")
    f.write(str(test_loss))
    f.close()

    samples_real(path = result_dir, data_loader=test_loader)

    plot_curve(result_dir, nll_val)

if __name__ == '__main__':

    main()