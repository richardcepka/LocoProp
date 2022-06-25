import torch
from torch import nn
import wandb

from wrapers import LocoPropWraper, FOOFWraper
from models import MLP
from fixed_point_solvers import fixed_point_iteration, anderson
from utils import acc, load_fashion_mnist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_config():
    cfg = dict(
        epochs=15,
        batch_size = 1024,

        hid_dim = 512,
        n_hid_layers = 5,
        nonlinearity = 'tanh',
        bias = True, 

        optimize='locoprop',
        iner_solver='fixed_point_iteration',
        f_type='identity',
        lambda_a=5000,
        lambda_r=100000,
        max_iter=10,

    )
    
    return cfg

def get_nonlinearity(cfg):
    if cfg.nonlinearity == 'tanh':
        return torch.tanh
    if cfg.nonlinearity == 'relu':
        return torch.relu
    if cfg.nonlinearity == 'softplus':
        return nn.functional.softplus
    
def get_iner_solver(cfg):
    if cfg.iner_solver == 'anderson':
        return anderson
    elif cfg.iner_solver == 'fixed_point_iteration':
        return fixed_point_iteration

def model2loco_prop(model, cfg):
    return LocoPropWraper(
        model=model, 
        iner_solver=get_iner_solver(cfg), 
        f_type=cfg.f_type,
        lambda_a=cfg.lambda_a, 
        lambda_r=cfg.lambda_r, 
        max_iter=cfg.max_iter
    )

def model2foof(model, cfg):
    return FOOFWraper(
        model=model, 
        learning_rate=cfg.learning_rate, 
        damping=cfg.damping, 
        inverse_step=cfg.inverse_step, 
        moving_step=cfg.moving_step,
    )


def train_epoch(wrapped_model, dataloader, criterion):
    runnig_loss = 0
    running_acc = 0 
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = wrapped_model(X)
        loss = criterion(y_pred, y)

        loss.backward()
        wrapped_model.step()
        wrapped_model.model.zero_grad() 

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
    if cfg.optimize == 'locoprop':
        wrapped_model = model2loco_prop(model, cfg)
    elif cfg.optimize == 'foof':
        wrapped_model = model2foof(model, cfg)

    criterion = nn.CrossEntropyLoss()

    for _ in range(cfg.epochs):
        metrics = train_epoch(wrapped_model, data['trainloader'], criterion)
        print(metrics)

if __name__ == "__main__":
    main()