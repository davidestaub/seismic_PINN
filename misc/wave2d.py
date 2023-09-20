import torch.utils
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)
import torch.nn as nn
import torch
import os
import random
import itertools

import wandb

torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)

# Constants:


velocity = 1
mu = (0.0, 0.0)
sigma = 0.12

test_tb_index = 0
only_init_percentage = 1


def lerp(y_value, yboundarypos1, yboundarypos2, ybounaryval1, yboundaryval2):
    return ybounaryval1 + (y_value - yboundarypos1) * (yboundaryval2 - ybounaryval1) / (yboundarypos2 - yboundarypos1)


class NeuralNet(nn.Module):

    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param,
                 regularization_exp, retrain_seed):
        super(NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function changed to softplus to match paper
        self.activation = nn.Tanh()
        self.regularization_param = regularization_param
        # Regularization exponent
        self.regularization_exp = regularization_exp
        # Random seed for weight initialization

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.retrain_seed = retrain_seed
        # Random Seed for weight initialization
        self.init_xavier()

    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network
        # (see equation above)
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss


class Pinns:
    def __init__(self, n_int_, n_tb_, n_tb_upsample_):
        self.n_int = n_int_
        self.n_tb = n_tb_
        self.n_tb_upsample = n_tb_upsample_

        # Extrema of the solution domain (t,x(x,y)) in [0,0.1]x[-1,1]
        self.domain_extrema = torch.tensor([[-1.0, 1.0],  # Time dimension
                                            [-1.0, 1.0], [-1.0, 1.0]])  # Space dimension

        # Number of space dimensions
        self.space_dimensions = 2

        # Parameter to balance role of data and PDE
        self.lambda_u = 2

        # FF Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=1,
                                              n_hidden_layers=3,
                                              neurons=64,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)

        wandb.watch(self.approximate_solution, log_freq=100)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_tb, self.training_set_tb2, self.training_set_int = self.assemble_datasets()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema

    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def upsample_convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        start = torch.tensor([self.domain_extrema[0, 0], mu[0] - 2.0 * sigma, mu[1] - 2.0 * sigma])
        end = torch.tensor([self.domain_extrema[0, 1], mu[0] + 2.0 * sigma, mu[1] + 2.0 * sigma])
        return tens * (end - start) + start

    def time_marching_convert(self, tens, t_start, t_end):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        start = torch.tensor([t_start, self.domain_extrema[1, 0], self.domain_extrema[2, 0]])
        end = torch.tensor([t_end, self.domain_extrema[1, 1], self.domain_extrema[2, 1]])
        return tens * (end - start) + start

    def get_layered_velocity(self, input):
        n_layers = 3
        layer_width = (abs(self.domain_extrema[2,0]) + abs(self.domain_extrema[2,1]))/n_layers
        smoothing_width = 0.1

        out = torch.zeros(input[:, 1].shape)
        for (index, element) in enumerate(out):
            y_value = input[index, 2]
            if (y_value < -0.4333):
                out[index] = 1.0

            elif (y_value > -0.4333 and y_value < -0.2333):
                out[index] = lerp(y_value, -0.4333, -0.2333, 1.0, -1.0)

            elif (y_value < 0.2333):
                out[index] = -1.0

            elif (y_value > 0.2333 and y_value < 0.4333):
                out[index] = lerp(y_value, 0.2333, 0.4333, -1.0, 1.0)

            else:
                out[index] = 1.0

        return out

    ################################################################################################

    # Function returning the input-output tensor required to assemble the training set S_tb corresponding to the temporal boundary
    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)

        # upsample:
        if not self.n_tb_upsample == 0:
            upsample_input_tb = self.upsample_convert(self.soboleng.draw(self.n_tb_upsample))
            upsample_input_tb[:, 0] = torch.full(upsample_input_tb[:, 0].shape, t0)

            input = torch.cat((input_tb, upsample_input_tb), 0)
            input_tb = input

        x_part = torch.pow(input_tb[:, 1] - mu[0], 2)
        y_part = torch.pow(input_tb[:, 2] - mu[1], 2)
        exponent = -0.5 * torch.pow((torch.sqrt(x_part + y_part) / sigma), 2)
        output = torch.exp(exponent)
        out = output.unsqueeze(dim=1)

        return input_tb, out

    def add_temporal_boundary_points_for_derivative(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)
        output_tb = torch.full((input_tb[:, 0].shape[0], 1), 0.0)
        return input_tb, output_tb

    #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):

        input_int = self.convert(self.soboleng.draw(self.n_int))
        '''This is what you do when you dont care about the output, I can use this when there is only a condition of TF but not on Ts'''
        output_int = torch.full((input_int.shape[0], 1), 0.0)
        return input_int, output_int

    def add_interior_points_time_marching(self, epoch, num_epochs):
        t_max = self.domain_extrema[0, 1]
        t_min = self.domain_extrema[0, 0]
        t_start = epoch / num_epochs * (t_max - t_min) + t_min
        t_end = (epoch + 1) / num_epochs * (t_max - t_min) + t_min
        t_end = min(t_end, t_max)

        # HOTFIX:
        t_start = self.domain_extrema[0, 0]

        input_int = self.time_marching_convert(self.soboleng.draw(self.n_int), t_start, t_end)

        print(input_int)

        # input_int = self.convert(self.soboleng.draw(self.n_int))
        '''This is what you do when you dont care about the output, I can use this when there is only a condition of TF but not on Ts'''
        output_int = torch.full((input_int.shape[0], 1), 0.0)
        return input_int, output_int

    def assemble_datasets_time_marching(self, epoch, num_epochs):
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_tb2, output_tb2 = self.add_temporal_boundary_points_for_derivative()
        input_int, output_int = self.add_interior_points_time_marching(epoch, num_epochs)  # S_int

        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb),
                                     batch_size=int((self.n_tb + self.n_tb_upsample)), shuffle=True, num_workers=4)
        training_set_tb2 = DataLoader(torch.utils.data.TensorDataset(input_tb2, output_tb2),
                                      batch_size=int(self.n_tb), shuffle=True, num_workers=4)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int),
                                      batch_size=int(self.n_int), shuffle=True, num_workers=4)

        return training_set_tb, training_set_tb2, training_set_int

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_tb2, output_tb2 = self.add_temporal_boundary_points_for_derivative()
        input_int, output_int = self.add_interior_points()  # S_int

        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb),
                                     batch_size=int((self.n_tb + self.n_tb_upsample)), shuffle=False, num_workers=4)
        training_set_tb2 = DataLoader(torch.utils.data.TensorDataset(input_tb2, output_tb2),
                                      batch_size=int(self.n_tb), shuffle=False, num_workers=4)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int),
                                      batch_size=int(self.n_int), shuffle=False, num_workers=4)

        return training_set_tb, training_set_tb2, training_set_int

    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual

    def apply_initial_condition(self, input_tb):
        u_pred_tb = self.approximate_solution(input_tb)
        return u_pred_tb

    # Function to compute the PDE residuals
    def compute_pde_residual1(self, input_int,velocity):
        u = self.approximate_solution(input_int)

        gradient = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]

        dt_gradient = torch.autograd.grad(gradient[:, 0].sum(), input_int, create_graph=True)[0]
        dx_gradient = torch.autograd.grad(gradient[:, 1].sum(), input_int, create_graph=True)[0]
        dy_gradient = torch.autograd.grad(gradient[:, 2].sum(), input_int, create_graph=True)[0]

        #rescaled_velocity = (((velocity +1.0)/(2.0)) * (2.943 - 1.323) + 1.323)

        residual1 = dx_gradient[:, 1] + dy_gradient[:, 2] - (1 / (torch.pow((((velocity +1.0)/(2.0)) * (2.943 - 1.323) + 1.323), 2))) * dt_gradient[:, 0]
        residual = residual1

        return residual.reshape(-1, )

    def compute_temporal_boundary_residual_for_derivative(self, inp_train_tb2, u_train_tb2):

        grad = torch.autograd.grad(self.approximate_solution(inp_train_tb2).sum(), inp_train_tb2, create_graph=True)[0]
        residual = u_train_tb2 - grad[:, 0]

        return residual

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_tb, u_train_tb, inp_train_tb2, u_train_tb2, inp_train_int,velocity, verbose=True):

        loss_tb2 = torch.mean(abs(self.compute_temporal_boundary_residual_for_derivative(inp_train_tb2, u_train_tb2)) ** 2)
        loss_int = torch.mean(abs(self.compute_pde_residual1(inp_train_int,velocity)) ** 2)

        loss_u = loss_tb + loss_tb2

        loss = torch.log10(25 * (loss_u) + loss_int)
        wandb.log({"loss": loss.item()})
        wandb.log({"BD loss": torch.log10(loss_u).item()})
        wandb.log({"PDE loss": torch.log10(loss_int).item()})
        # loss = loss_tb
        #if verbose: print("Total loss: ", round(loss.item(), 4), "| BD Loss: ", round(torch.log10(loss_u).item(), 4),
                        #  "| PDE Loss: ", round(torch.log10(loss_int).item(), 4))

        return loss

    def compute_loss_init(self, inp_train_tb, u_train_tb, inp_train_tb2, u_train_tb2, verbose=True):

        u_pred_tb = self.apply_initial_condition(inp_train_tb)

        assert (u_pred_tb.shape[1] == u_train_tb.shape[1])

        r_tb = u_train_tb - u_pred_tb

        r_tb2 = self.compute_temporal_boundary_residual_for_derivative(inp_train_tb2, u_train_tb2)

        loss_tb = torch.mean(abs(r_tb) ** 2)
        loss_tb2 = torch.mean(abs(r_tb2) ** 2)

        loss_u = loss_tb + loss_tb2

        loss = torch.log10(self.lambda_u * (loss_u))
        wandb.log({"loss": loss})
        # loss = loss_tb
        if verbose: print("Total loss: ", round(loss.item(), 4), "| BD Loss: ", round(torch.log10(loss_u).item(), 4),
                          "| PDE Loss: ", "N.A")

        return loss

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=False):

        inp_train_tb, u_train_tb = next(iter(self.training_set_tb))
        inp_train_tb2, u_train_tb2 = next(iter(self.training_set_tb2))
        inp_train_int, u_train_int = next(iter(self.training_set_int))

        inp_train_tb = inp_train_tb.to(device)
        u_train_tb = u_train_tb.to(device)
        inp_train_tb2 = inp_train_tb2.to(device)
        u_train_tb2 = u_train_tb2.to(device)
        inp_train_int = inp_train_int.to(device)
        velocity = self.get_layered_velocity(inp_train_int)
        print(inp_train_tb,inp_train_int,velocity)

        velocity = velocity.to(device)
        self.approximate_solution = self.approximate_solution.to(device)

        print("ON GPU?", velocity.is_cuda)
        print("ON GPU?", inp_train_tb.is_cuda)
        print("ON GPU?", inp_train_tb2.is_cuda)
        print("ON GPU?", u_train_tb.is_cuda)
        print("ON GPU?", u_train_tb2.is_cuda)
        print("ON GPU?", inp_train_int.is_cuda)
        print("ON GPU?", next(self.approximate_solution.parameters()).is_cuda)

        inp_train_tb2.requires_grad = True
        inp_train_int.requires_grad = True

        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            def closure():

                optimizer.zero_grad()

                if epoch <= (only_init_percentage / 100) * num_epochs:
                    loss = self.compute_loss_init(inp_train_tb, u_train_tb, inp_train_tb2, u_train_tb2,verbose=verbose)
                else:
                    loss = self.compute_loss(inp_train_tb, u_train_tb, inp_train_tb2, u_train_tb2, inp_train_int,velocity,verbose=verbose)
                loss.backward()

                history.append(loss.item())
                return loss

            optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])

        return history

    ################################################################################################

#marmousi = np.fromfile('vel_nx850_nz156_dx20.dat', dtype="byte")
#print("marmousi = ",marmousi.shape)

n_int = 35000
n_tb = n_int
n_tb_upsample = 0

wandb.init(project='test',name = 'report 3 layer unconditioned normalized showuld match prev')


pinn = Pinns(n_int, n_tb, n_tb_upsample)


n_epochs = 4000
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                              lr=float(0.5),
                              max_iter=100,
                              max_eval=100,
                              history_size=2000,
                              line_search_fn="strong_wolfe",
                              tolerance_grad=1e-8,
                              tolerance_change=1.0 * np.finfo(float).eps)

hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_LBFGS,
                verbose=True)


torch.save(pinn.approximate_solution.state_dict(),
           "report_unconditioned_normalized_3layers_to_match_experiment.pth")

