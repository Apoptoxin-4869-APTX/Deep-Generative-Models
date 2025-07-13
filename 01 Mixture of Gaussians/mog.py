# inspired by Jakub M. Tomczak
# written by Apoptoxin-4869

# importing necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt

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


class MoG(nn.Module):
    def __init__(self, D, K, uniform=False):
        super(MoG, self).__init__()

        print('MoG by JT.')
        
        # hyperparams
        self.uniform = uniform
        self.D = D  # the dimensionality of the input
        self.K = K  # the number of components
        
        # params
        self.mu = nn.Parameter(torch.randn(1, self.K, self.D) * 0.25 + 0.5)
        self.log_var = nn.Parameter(-3. * torch.ones(1, self.K, self.D))
        
        if self.uniform:
            self.w = torch.zeros(1, self.K)
            self.w.requires_grad = False
        else:
            self.w = nn.Parameter(torch.zeros(1, self.K))

        # other
        self.PI = torch.from_numpy(np.asarray(np.pi))
    
    def log_diag_normal(self, x, mu, log_var, reduction='sum', dim=1):
        log_p = -0.5 * torch.log(2. * self.PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x.unsqueeze(1) - mu)**2.
        return log_p
    
    def forward(self, x, reduction='mean'):
        # calculate components
        log_pi = torch.log(F.softmax(self.w, 1))  # B x K, softmax is used for R^K -> [0,1]^K s.t. sum(pi) = 1
        log_N = torch.sum(self.log_diag_normal(x, self.mu, self.log_var), 2)  # B x K, log-diag-Normal for K components
        # =====LOSS: Negative Log-Likelihood
        NLL_loss = -torch.logsumexp(log_pi + log_N,  1)  # B
        
        # Final LOSS
        if reduction == 'sum':
            return NLL_loss.sum()
        elif reduction == 'mean':
            return NLL_loss.mean()
        else:
            raise ValueError('Either `sum` or `mean`.')

    def sample(self, batch_size=64):
        # init an empty tensor
        x_sample = torch.empty(batch_size, self.D)
        
        # sample components
        pi = F.softmax(self.w, 1)  # B x K, softmax is used for R^K -> [0,1]^K s.t. sum(pi) = 1
                             
        indices = torch.multinomial(pi, batch_size, replacement=True).squeeze()
        
        for n in range(batch_size):
            indx = indices[n]  # pick the n-th component
            x_sample[n] = self.mu[0,indx] + torch.exp(0.5*self.log_var[0,indx]) * torch.randn(self.D)
        
        return x_sample
    
    def log_prob(self, x, reduction='mean'):
        with torch.no_grad():
            # calculate components
            log_pi = torch.log(F.softmax(self.w, 1))  # B x K, softmax is used for R^K -> [0,1]^K s.t. sum(pi) = 1
            log_N = torch.sum(self.log_diag_normal(x, self.mu, self.log_var), 2)  # B x K, log-diag-Normal for K components
        
            # log_prob
            log_prob = torch.logsumexp(log_pi + log_N,  1)  # B
            
            if reduction == 'sum':
                return log_prob.sum()
            elif reduction == 'mean':
                return log_prob.mean()
            else:
                raise ValueError('Either `sum` or `mean`.')
            


def evaluation(test_loader, name=None, model_best=None, epoch=None):
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(name + '.model')

    model_best.eval()
    loss = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(test_loader):
        loss_t = -model_best.log_prob(test_batch, reduction='sum')
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N

    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')

    return loss


def samples_real(name, test_loader):
    # REAL-------
    num_x = 4
    num_y = 4
    x = next(iter(test_loader)).detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name+'_real_images.png', bbox_inches='tight')
    plt.close()


def samples_generated(name, data_loader, extra_name=''):
    with torch.no_grad():
        # GENERATIONS-------
        model_best = torch.load(name + '.model')

        num_x = 4
        num_y = 4
        x = model_best.sample(batch_size=num_x * num_y)
        x = x.detach().numpy()

        fig, ax = plt.subplots(num_x, num_y)
        for i, ax in enumerate(ax.flatten()):
            plottable_image = np.reshape(x[i], (8, 8))
            ax.imshow(plottable_image, cmap='gray')
            ax.axis('off')

        plt.savefig(name + '_generated_images' + extra_name + '.png', bbox_inches='tight')
        plt.close()

def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + '_nll_val_curve.png', bbox_inches='tight')
    plt.close()

def means_save(name, extra_name='', num_x = 4, num_y = 4):
    with torch.no_grad():
        # GENERATIONS-------
        model_best = torch.load(name + '.model')

        pi = F.softmax(model_best.w, 1).squeeze()

        x = model_best.mu[:, 0:num_x * num_y]
        N = x.shape[1]
        x = x.squeeze(0).detach().numpy()

        fig, ax = plt.subplots(int(np.sqrt(N)), int(np.sqrt(N)))
        for i, ax in enumerate(ax.flatten()):
            plottable_image = np.reshape(x[i], (8, 8))
            ax.imshow(plottable_image, cmap='gray')
            ax.set_title(f'$\pi$ = {pi[i].item():.5f}')
            ax.axis('off')
        fig.tight_layout()
        plt.savefig(name + '_means_images' + extra_name + '.png', bbox_inches='tight')
        plt.close()


def training(name, max_patience, num_epochs, model, optimizer, training_loader, val_loader):
    nll_val = []
    best_nll = 1000.
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for indx_batch, batch in enumerate(training_loader):
            loss = model.forward(batch)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        loss_val = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting

        if e == 0:
            print('saved!')
            torch.save(model, name + '.model')
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print('saved!')
                torch.save(model, name + '.model')
                best_nll = loss_val
                patience = 0
            else:
                patience = patience + 1
        
        samples_generated(name, val_loader, extra_name="_epoch_" + str(e))
        
        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)

    return nll_val


def main():
    # EXPERIMENTS

    # changing to [-1, 1] and adding small Gaussian noise
    transforms = tt.Lambda(lambda x: (x/17.) + (np.random.randn(*x.shape)/136.))

    train_data = Digits(mode='train', transforms=transforms)
    val_data = Digits(mode='val', transforms=transforms)
    test_data = Digits(mode='test', transforms=transforms)

    training_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


    # hyperparameters
    D = 64   # input dimension

    K = 50  # the number of neurons in scale (s) and translation (t) nets

    lr = 1e-3 # learning rate
    num_epochs = 2000 # max. number of epochs
    max_patience = 100 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped

    # create result directory 
    name = 'mog' + '_' + str(K)
    if not (os.path.exists('results/')):
        os.mkdir('results/')
    result_dir = 'results/' + name + '/'
    if not (os.path.exists(result_dir)):
        os.mkdir(result_dir)

    # intialize model
    model = MoG(D=D, K=K, uniform=True)

    # OPTIMIZER
    # optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad == True], lr=lr, momentum=0.1, weight_decay=1.e-4)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad == True], lr=lr)

    # Training procedure
    nll_val = training(name=result_dir + name, 
                    max_patience=max_patience, 
                    num_epochs=num_epochs, 
                    model=model, 
                    optimizer=optimizer,
                    training_loader=training_loader, 
                    val_loader=val_loader)

    # Evaluation
    test_loss = evaluation(name=result_dir + name, test_loader=test_loader)
    f = open(result_dir + name + '_test_loss.txt', "w")
    f.write(str(test_loss))
    f.close()

    samples_real(result_dir + name, test_loader)
    samples_generated(result_dir + name, test_loader, extra_name='FINAL')

    means_save(result_dir + name, extra_name='_'+str(K), num_x=5, num_y=5)

    plot_curve(result_dir + name, nll_val)

if __name__ == '__main__':

    main()