import torch
from torch import nn
import wandb

from models import MLP
from utils import acc, load_fashion_mnist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_config():
    cfg = dict(
        epochs=15,
        batch_size=1024,

        hid_dim=512,
        n_hid_layers=5,
        nonlinearity='tanh',
        bias=True, 

        opt='adam',
        learning_rate=0.001

    )
    
    return cfg

def get_nonlinearity(cfg):
    if cfg.nonlinearity == 'tanh':
        return torch.tanh
    if cfg.nonlinearity == 'relu':
        return torch.relu
    if cfg.nonlinearity == 'softplus':
        return nn.functional.softplus

def build_optimizer(cfg, params):
    # K-FAC https://github.com/Thrandis/EKFAC-pytorch/blob/master/kfac.py
    # Shampoo: https://github.com/facebookresearch/optimizers/tree/main/shampoo
    
    if cfg['opt'] == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=cfg['learning_rate'],
        )

    elif cfg['opt'] == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=cfg['learning_rate'],
        )

    elif cfg['opt'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            params,
            lr=cfg['learning_rate'],
        )


    return optimizer

def train_epoch(model, optimizer, dataloader, criterion):
    runnig_loss = 0
    running_acc = 0 
    for X, y in dataloader:
        optimizer.zero_grad()

        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()

        wandb.log({'train_loss': loss.item()})
        runnig_loss += loss.item() 
        running_acc += acc(y_pred, y)

    return dict(
        train_loss=runnig_loss/len(dataloader),
        train_acc=running_acc/len(dataloader)
    )

def main():
    wandb.init(config=get_config(), project='LocoProp')
    cfg = wandb.config

    data = load_fashion_mnist(batch_size=cfg.batch_size)
    model = MLP(
        in_dim=data['dim'], hid_dim=cfg.hid_dim, out_dim=data['num_classes'], 
        n_hid_layers=cfg.n_hid_layers, nonlinearity=get_nonlinearity(cfg), bias=cfg.bias
    ).to(device)
    optimizer = build_optimizer(cfg, model.parameters())
    criterion = nn.CrossEntropyLoss()

    for _ in range(cfg.epochs):
        metrics = train_epoch(model, optimizer, data['trainloader'], criterion)
        print(metrics)

if __name__ == "__main__":
    main()