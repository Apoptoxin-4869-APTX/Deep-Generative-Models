# main author Jakub Tomczak
# modified by Ertunc Erdil
# further modified by Apoptoxin-4869

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

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

        self.max_intensity = 16.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx] / self.max_intensity

        if self.transforms:
            sample = self.transforms(sample)
        return sample

# === Define CausalConv1d class === #
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, mask_type=False, **kwargs):
        super(CausalConv1d, self).__init__()

        # Store configuration parameters.
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.mask_type = mask_type
        
        # Calculate the left-side padding needed to enforce causality.
        # The standard padding is (kernel_size - 1) * dilation, which ensures that the output at time n
        # can use inputs from time n - (kernel_size - 1) * dilation up to time n.
        # If A is True (mask A mode), we add an extra padding unit so that the output at time n
        # depends only on inputs from times strictly less than n (x_{<n}), effectively shifting the receptive field.
        self.padding = (kernel_size - 1) * dilation + (1 if mask_type else 0)

        # Create the Conv1d module without internal padding.
        # We will manually apply left-padding in the forward() method.
        # Note: PyTorch’s Conv1d applies symmetric padding when using the 'padding' parameter,
        # but for autoregressive (causal) convolutions, we require padding only on the left.
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,  # Padding is applied manually
            dilation=dilation,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Apply left-padding manually (pad_left, pad_right). Only the left side is padded.
        x = F.pad(x, (self.padding, 0))
        conv1d_out = self.conv1d(x)
        if self.mask_type:
            # In mask A mode, drop the final output time step because its receptive field includes x_n.
            return conv1d_out[:, :, :-1]
        else:
            # In the standard mode, keep the full output since the receptive field is allowed to include x_n.
            return conv1d_out

# === Define ARM class === #
class ARM(nn.Module):
    def __init__(self, net, input_dims=2, num_outputs=256):
        super(ARM, self).__init__()
        """
        Initialize ARM model
        
        Args:
            net (torch.nn.modules.container.Sequential): Network
            input_dims (int): Dimensionality of the input data (D)
            num_outputs (int): Number of output units in the network which corresponds to the number of possible intensity values, K.
        """

        self.net = net
        self.num_outputs = num_outputs
        self.input_dims = input_dims

    def f(self, x):
        """
        Performs forward pass in the network and computes output probabilities.
        Args:
            x (torch.Tensor): Input data of shape (N, D)

        Return:
            prob (torch.Tensor): Predicted probabilities for each data point N and each class K. Shape (N, D, K).
        """
        x = x.unsqueeze(1) # Add channel dimension C = 1 since the images have single channel. Shape (N, 1, K) 
        h = self.net(x) # Output of the network before softmax. Shape (N, K, D)

        h = h.permute(0, 2, 1) # Take the class dimension to the end. Shape (N, D, K)
        prob = torch.softmax(h, 2) # Apply softmax along the class dimension to get probabilities. Shape (N, D, K)

        return prob
        
    def forward(self, x, reduction='avg'):
        """
        Forward function that performs forward pass and compute the NLL loss.
        Args:
            x (torch.Tensor): Input data of shape (N, D)

        Return:
            NLL_loss (torch.Tensor): Negative log-likelihood loss of shape ()
        """
        NLL_loss = -self.log_prob(x) # Negative log probability for each sample N. Shape (N)

        if reduction == 'avg':
            NLL_loss = NLL_loss.mean() # Average over the dimension N. Shape ()
        elif reduction == 'sum':
            NLL_loss = NLL_loss.sum() # Sum over the dimension N. Shape ()
        else:
            raise ValueError('reduction could be either `avg` or `sum`.')
        
        return NLL_loss

    def log_prob(self, x):
        """
        Performs forward pass and return log categorical distribution values for each sample N
        Args:
            x (torch.Tensor): Input data of shape (N, D)

        Return:
            log_p (torch.Tensor): Negative log-likelihood loss of shape (N, )
        """
        prob = self.f(x) # Predicted probabilities for each data point N and each class K. Shape (N, D, K).

        log_p = log_categorical(x, prob, num_classes=self.num_outputs, reduction='sum') # Return log Categorical by summing over all classes. Shape (N, D)        
        log_p = log_p.sum(-1) # Sum over the dimension D. Shape (N)

        return log_p

    def sample(self, n_samples):
        """
        Generate samples from the model
        Args:
            n_samples (int): Number of samples to be generated

        Return:
            samples (torch.Tensor): Generated samples (n_samples, D)
        """
        samples = torch.zeros((n_samples, self.input_dims))

        for d in range(self.input_dims):
            p = self.f(samples) # Shape (n_samples, D, K)
            sample_d = torch.multinomial(p[:, d, :], num_samples=1) # Only sample from the d^th dimension at every iteration. Shape (n_samples, 1)
            
            samples[:, d] = sample_d[:, 0] / (self.num_outputs - 1)

        return samples * (self.num_outputs - 1)

# === Function to compute log Categorical distribution === #
def log_categorical(x, prob, num_classes=256, reduction=None):
    """
    Compute log Categorical distribution, i.e., log Cat(x_{n, d}|x_{n, <d})
    Args:
        x (torch.Tensor): Input data of shape (N, D)
        prob (torch.Tensor): Predicted probabilities for each data point N and each class K. Shape (N, D, K).
        num_classes (int): Number of possible intensity values, K.
        reduction (str): 'avg' or 'sum'

    Return:
        log_p (torch.Tensor): log Categorical distribution. Shape (N, D)
    """
    EPS = 1.e-5
    x_one_hot = F.one_hot((x*(num_classes-1)).long(), num_classes=num_classes) # Convert x to one-hot representation. Shape (N, D, K)
    log_p = x_one_hot * torch.log(torch.clamp(prob, EPS, 1. - EPS)) # log Categorical for each data N, each dimension D, and each class K (N, D, K)

    if reduction == 'avg':
        return torch.mean(log_p, -1)
    elif reduction == 'sum':
        return torch.sum(log_p, -1)
    else:
        return log_p

# === Function to evaluate the model === #
def evaluation(data_loader, model_path=None, model=None, epoch=None):
    """
    Evaluate the model
    Args:
        data_loader (torch.utils.data.dataloader.DataLoader): Data loader to be used for evaluation
        model_path (str): The path to the saved pretrained model
        model (ARM): The ARM object containing the model
        epoch (int): Epoch number at which the model is evaluated

    Return:
        loss (float): Average loss on the evaluation set
    """
    if model is None:
        # load best performing model
        model = torch.load(f"{model_path}/arm.model", weights_only=False)

    model.eval()
    loss = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(data_loader):
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
    # x = next(iter(data_loader)).detach().numpy()

    model = torch.load(f"{path}/arm.model", weights_only=False)
    model.eval()

    num_x = 4
    num_y = 4
    x = model.sample(num_x * num_y)
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
def training(path, num_outputs, max_patience, num_epochs, model, optimizer, train_loader, val_loader):
    """
    Function to train the model
    Args:
        path (str): The path to save the outputs
        num_outputs (int): Number of output units in the network or number of intensity values that a pixel can take.
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
            
            loss = model.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        loss_val = evaluation(data_loader=val_loader, model=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting

        if e == 0:
            print('saved!')
            torch.save(model, f"{path}/arm.model")
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print('saved!')
                torch.save(model, f"{path}/arm.model")
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
    input_dims = 64 # Input dimension of the data

    lr = 1e-3 # Learning rate
    num_epochs = 100 # Max. number of epochs for training
    max_patience = 20 # If training doesn't improve for longer than max_patience epochs, it is stopped
    
    # Number of output channels in the intermediate convolutional layers.
    out_channels = 256  
    
    # Number of outputs in the final layer. 
    # Note that the intensities range between 0-16 in the Digits dataset. 
    # Since there are 17 possible values, we set num_outputs=17
    num_outputs = 17 

    k_size = 3 # kernel size of convolutional layers
    batch_size = 32 # batch size

    # === Create dataset objects and dataloaders === #
    train_data = Digits(mode='train')
    val_data = Digits(mode='val')
    test_data = Digits(mode='test')

    training_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Run the following lines to save an example image from the dataset.
    # img = next(iter(training_loader))
    # plt.imsave('image.png', img[0].reshape(8, 8))

    # === Define the network === #
    net = nn.Sequential(
        # Note that A=True in the first layer and A=false in the following layers
        CausalConv1d(in_channels=1, out_channels=out_channels, dilation=1, kernel_size=k_size, mask_type=True, bias=True),
        nn.LeakyReLU(),
        CausalConv1d(in_channels=out_channels, out_channels=out_channels, dilation=1, kernel_size=k_size, mask_type=False, bias=True),
        nn.LeakyReLU(),
        CausalConv1d(in_channels=out_channels, out_channels=out_channels, dilation=1, kernel_size=k_size, mask_type=False, bias=True),
        nn.LeakyReLU(),
        CausalConv1d(in_channels=out_channels, out_channels=num_outputs, dilation=1, kernel_size=k_size, mask_type=False, bias=True))

    # === Create ARM object === #
    model = ARM(net, input_dims=input_dims, num_outputs=num_outputs)

    result_dir = './results_1d_causal_conv/'
    if not(os.path.exists(result_dir)):
        os.mkdir(result_dir)

    # Define optimizer
    optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)

    # Start training
    nll_val = training(path=result_dir,
                       num_outputs=num_outputs,
                       max_patience=max_patience, 
                       num_epochs=num_epochs, 
                       model=model, 
                       optimizer=optimizer, 
                       train_loader=training_loader, 
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