import configparser
import numpy as np
import torch.optim as optim
import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(2)
import torch
import os
import wandb
from misc import PINN_model

torch.set_printoptions(threshold=10000)
torch.set_printoptions(linewidth=200)

config = configparser.ConfigParser()
config.read("config.ini")


torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
    config['description']['tag'],
    config['Network']['n_points'],
    config['optimizer']['n_epochs'],
    config['optimizer']['lr'],
    config['optimizer']['max_iter'],
    config['optimizer']['max_eval'],
    config['optimizer']['history_size'],
    config['Network']['n_hidden_layers'],
    config['Network']['n_neurons'],
    config['Network']['activation']
    ,config['initial_condition']['a']
    ,config['initial_condition']['b']
)

wandb.init(project='Semester Thesis',name = name)
pinn = PINN_model.Pinns(int(config['Network']['n_points']))
n_epochs = int(config['optimizer']['n_epochs'])
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                                lr=float(float(config['optimizer']['lr'])),
                                max_iter=int(config['optimizer']['max_iter']),
                                max_eval=int(config['optimizer']['max_eval']),
                                history_size=int(config['optimizer']['history_size']),
                                line_search_fn="strong_wolfe",
                                tolerance_grad=1e-8,
                                tolerance_change=1.0 * np.finfo(float).eps)

hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_LBFGS,
                verbose=True)
torch.save(pinn.approximate_solution.state_dict(),name+'.pth')
