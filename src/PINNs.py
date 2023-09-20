import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
torch.manual_seed(2)
import torch.nn as nn
import wandb
import mixture_model
import torch
from torch.utils.data import DataLoader
torch.manual_seed(128)
import os
import initial_conditions
import numpy as np
import torch
import FD_devito
from devito import *
import pickle




torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class NeuralNet(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param,
                 regularization_exp, retrain_seed,activation):
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
        #if config['Network']['activation'] == 'tanh':
            #self.activation = nn.Tanh()
        self.activation = activation
        #else:
            #print("unknown activation function", config['Network'].activation)
            #exit()
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
                m.bias.data.fill_(0.01)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss

class Pinns:

    def __init__(self, n_collocation_points,wandb_on,config):
        if config['Network']['activation'] == 'tanh':
             self.activation = nn.Tanh()

        else:
            print("unknown activation function", config['Network'].activation)
            exit()
        self.n_collocation_points = n_collocation_points
        self.wandb_on = wandb_on

        self.domain_extrema = torch.tensor([[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
                                                [float(config['domain']['xmin']), float(config['domain']['xmax'])], [float(config['domain']['ymin']), float(config['domain']['ymax'])],
                                                ])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=2,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3,activation=self.activation)
        if wandb_on:
            wandb.watch(self.approximate_solution, log_freq=100)


        if config['initial_condition']['source_function'] == 'explosion':
            self.source_function = initial_conditions.initial_condition_explosion
        elif config['initial_condition']['source_function'] == 'explosion_conditioned':
            self.source_function = initial_conditions.initial_condition_explosion_conditioned
        elif config['initial_condition']['source_function'] == 'explosion_two_sources':
            self.source_function = initial_conditions.initial_condition_explosion_two_sources
        elif config['initial_condition']['source_function'] == 'gaussian':
            self.source_function = initial_conditions.initial_condition_gaussian
        elif config['initial_condition']['source_function'] == 'donut':
            self.source_function = initial_conditions.initial_condition_donut
        else:
            print(config['initial_condition']['source_function'],'explosion', 'explosion' ==config['initial_condition']['source_function'])
            raise Exception("Source function {} is not implemented".format(config['initial_condition']['source_function']))




        self.t1 = float(config['initial_condition']['t1'])
        self.t1 = torch.tensor(self.t1)
        self.t1 = self.t1.to(device)
        self.sigma_quake = float(config['parameters']['sigma_quake'])
        self.sigma_quake = torch.tensor(self.sigma_quake)
        self.sigma_quake = self.sigma_quake.to(device)
        self.rho_solid = float(config['parameters']['rho_solid'])
        self.rho_solid = torch.tensor(self.rho_solid)
        self.rho_solid = self.rho_solid.to(device)
        self.parameter_model = config['parameters']['model_type']
        self.lambda_solid = config['parameters']['lambda_solid']
        self.mu_solid = config['parameters']['mu_solid']

        self.visualize = config['visualize']['visualize_on']
        self.test_on = config['test']['test_on']


    def pinn_model_eval(self, input_tensor):
        # Evaluate the model with the given input tensor
        U_perturbation = self.approximate_solution(input_tensor)
        #t = input_tensor[:, 0]
        #u0x, u0y = initial_conditions.initial_condition_explosion_conditioned(input_tensor)
        u0x, u0y = self.source_function(input_tensor,self.sigma_quake)
        U = torch.zeros_like(U_perturbation)
        #t1 = float(config['initial_condition']['t1'])
        U[:, 0] = U_perturbation[:, 0] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0x * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2)
        U[:, 1] = U_perturbation[:, 1] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0y * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2)
        return U

    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
            print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))

        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid ))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid ))
            print(lambda_m.shape,mu_m.shape)
            print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))

        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
            print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))

        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        return input_s

    def assemble_datasets(self):
        input_s1 = self.add_solid_points()

        training_set_s1 = DataLoader(torch.utils.data.TensorDataset(input_s1),
                                     batch_size=int(self.n_collocation_points), shuffle=False, num_workers=4)
        return training_set_s1

    def compute_solid_loss(self, input_s):

        U = self.pinn_model_eval(input_s)

        #u_x = U[:, 0].unsqueeze(1)
        #u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(U[:, 0].unsqueeze(1).sum(), input_s, create_graph=True)[0]

        gradient_y = torch.autograd.grad(U[:, 1].unsqueeze(1).sum(), input_s, create_graph=True)[0]

        #dt_x = gradient_x[:, 0]
        #dx_x = gradient_x[:, 1]
        #dy_x = gradient_x[:, 2]
        #dt_y = gradient_y[:, 0]
        #dx_y = gradient_y[:, 1]
        #dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(gradient_x[:, 0].sum(), input_s, create_graph=True)[0][:, 0]

        dt2_y = torch.autograd.grad(gradient_y[:, 0].sum(), input_s, create_graph=True)[0][:, 0]

        #diag_1 = 2.0 * dx_x
        #diag_2 = 2.0 * dy_y
        #off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((2.0 * gradient_x[:, 1], gradient_x[:, 2] + gradient_y[:, 1])), torch.stack((gradient_x[:, 2] + gradient_y[:, 1], 2.0 * gradient_y[:, 2]))), dim=1)


        stress_tensor_00 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[0, 0]

        stress_tensor_off_diag = 2.0 * self.mu_m * eps[0, 1]

        stress_tensor_11 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[1, 1]

        #stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     #torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)
        off_diag_grad = torch.autograd.grad(stress_tensor_off_diag.sum(), input_s, create_graph=True)[0]


        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)

        div_stress[0, :] = torch.autograd.grad(stress_tensor_00.sum(), input_s, create_graph=True)[0][:, 1] + \
                           off_diag_grad[:, 2]

        div_stress[1, :] = off_diag_grad[:, 1] + \
                           torch.autograd.grad(stress_tensor_11.sum(), input_s, create_graph=True)[0][:, 2]


        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)

        residual_solid = self.rho_solid * dt2_combined - div_stress

        residual_solid = residual_solid.reshape(-1, )


        loss_solid = torch.mean(abs(residual_solid) ** 2)


        return loss_solid

    def get_solid_residual(self, input_s):
        U = self.pinn_model_eval(input_s)
        u_x = U[:, 0].unsqueeze(1)
        u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(u_x.sum(), input_s, create_graph=True)[0]
        gradient_y = torch.autograd.grad(u_y.sum(), input_s, create_graph=True)[0]
        dt_x = gradient_x[:, 0]
        dx_x = gradient_x[:, 1]
        dy_x = gradient_x[:, 2]
        dt_y = gradient_y[:, 0]
        dx_y = gradient_y[:, 1]
        dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(dt_x.sum(), input_s, create_graph=True)[0][:, 0]
        dt2_y = torch.autograd.grad(dt_y.sum(), input_s, create_graph=True)[0][:, 0]

        diag_1 = 2.0 * dx_x
        diag_2 = 2.0 * dy_y
        off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((diag_1, off_diag)), torch.stack((off_diag, diag_2))), dim=1)

        stress_tensor_00 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[0, 0]
        stress_tensor_off_diag = 2.0 * self.mu_m * eps[0, 1]
        stress_tensor_11 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[1, 1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        div_stress[0, :] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[1, :] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]

        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        residual_solid = self.rho_solid * dt2_combined - div_stress
        residual_solid = residual_solid.reshape(-1, )

        U_without_ansatz = self.approximate_solution(input_s)

        return residual_solid, U, U_without_ansatz

    def compute_loss(self, inp_train_s):
        loss_solid = self.compute_solid_loss(inp_train_s)
        loss = torch.log10(loss_solid)
        if self.wandb_on:
            wandb.log({"loss": loss.item()})
            wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        return loss

    def compute_test_loss(self, test_input, mu_quake):

        # Number of points along x and y axis, total number of points will be numpoints_sqrt**2
        numpoints_sqrt = 256
        print("start")

        res_list_ux = []
        res_list_uy = []
        res_list_u = []
        time_list = np.linspace(0, 1, 101).tolist()
        for i in time_list:

            time = i
            inputs = test_input
            inputs[:, 0] = time
            inputs = inputs.to(device)

            ux = self.pinn_model_eval(inputs)[:, 0]
            uy = self.pinn_model_eval(inputs)[:, 1]
            ux_out = ux.detach()
            uy_out = uy.detach()

            np_ux_out = ux_out.cpu().numpy()
            np_uy_out = uy_out.cpu().numpy()

            B_ux = np.reshape(np_ux_out, (-1, int(np.sqrt(np_ux_out.shape[0]))))
            B_uy = np.reshape(np_uy_out, (-1, int(np.sqrt(np_uy_out.shape[0]))))
            B = np.sqrt(B_uy ** 2 + B_ux ** 2)
            res_list_ux.append(B_ux)
            res_list_uy.append(B_uy)
            res_list_u.append(B)

        res_ux = np.dstack(res_list_ux)
        res_uy = np.dstack(res_list_uy)
        res_ux = np.rollaxis(res_ux, -1)
        res_uy = np.rollaxis(res_uy, -1)
        res_u = np.dstack(res_list_u)
        res_u = np.rollaxis(res_u, -1)
        s_ux = 5 * np.mean(np.abs(res_uy))
        s_uy = 5 * np.mean(np.abs(res_uy))
        s_u = 5 * np.mean(np.abs(res_u))


        f, axarr = plt.subplots(3, 3, figsize=(15, 20))
        plt.subplots_adjust(hspace=-0.1, wspace=0.1)
        test_loss = 0
        count_after_decimal_x = str(mu_quake[0])[::-1].find('.')
        count_after_decimal_y = str(mu_quake[1])[::-1].find('.')
        if count_after_decimal_x > 3:
            mu_quake_str_x = str(round(float(mu_quake[0]), 3))
        else:
            mu_quake_str_x = str(float(mu_quake[0]))

        if count_after_decimal_y > 3:
            mu_quake_str_y = str(round(float(mu_quake[1]), 3))
        else:
            mu_quake_str_y = str(float(mu_quake[1]))

        file_name_x = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_x.pkl'.format(self.parameter_model, mu_quake_str_x,mu_quake_str_y)
        file_name_y = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_y.pkl'.format(self.parameter_model, mu_quake_str_x,mu_quake_str_y)
        file_name_u = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_u.pkl'.format(self.parameter_model, mu_quake_str_x,mu_quake_str_y)

        with open(file_name_x, 'rb') as f_:
            res_list_devito_x = pickle.load(f_)
        with open(file_name_y, 'rb') as f_:
            res_list_devito_y = pickle.load(f_)
        with open(file_name_u, 'rb') as f_:
            res_list_devito_u = pickle.load(f_)

        s = 5 * np.mean((np.abs(res_list_devito_x[0])))



        for h in range(0, len(res_list_uy)):
            diffu = ((res_u[h, :, :]) - (res_list_devito_u[h]))
            diffu[0:9, :] = 0
            diffu[:, 0:9] = 0
            diffu[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
            diffu[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
            test_loss = test_loss + np.sqrt(np.mean(diffu**2))


            if h == 0 or h == int(len(res_list_uy) / 4) or h == int(len(res_list_uy) / 3) or h == int(
                    len(res_list_uy) / 2) or h == len(res_list_uy) - 2:
                diffx = ((res_uy[h, :, :]) - (res_list_devito_x[h]))
                diffx[0:9, :] = 0
                diffx[:, 0:9] = 0
                diffx[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffx[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1x = axarr[0][0].imshow(res_uy[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2x = axarr[0][1].imshow(res_list_devito_x[h], 'bwr', vmin=-s, vmax=s)
                im3x = axarr[0][2].imshow(diffx, 'bwr', vmin=-s, vmax=s)

                diffy = (res_ux[h, :, :]) - (res_list_devito_y[h])
                diffy[0:9, :] = 0
                diffy[:, 0:9] = 0
                diffy[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffy[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1y = axarr[1][0].imshow((res_ux[h, :, :]), 'bwr', vmin=-s, vmax=s)
                im2y = axarr[1][1].imshow((res_list_devito_y[h]), 'bwr', vmin=-s, vmax=s)
                im3y = axarr[1][2].imshow(diffy, 'bwr', vmin=-s, vmax=s)


                im1u = axarr[2][0].imshow(res_u[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2u = axarr[2][1].imshow(res_list_devito_u[h], 'bwr', vmin=-s, vmax=s)
                im3u = axarr[2][2].imshow(diffu, 'bwr', vmin=-s, vmax=s)


                axarr[0][0].set_title("PINN", fontsize=25, pad=20)
                axarr[0][1].set_title("Devito", fontsize=25, pad=20)
                axarr[0][2].set_title("Difference", fontsize=25, pad=20)
                f.subplots_adjust(right=0.8)
                cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                f.colorbar(im3x, cax=cbar_ax)

                if self.wandb_on:
                    wandb.log({"Test set difference @ time = {}".format(h): wandb.Image(f)})


        test_loss = test_loss / len(res_list_uy)
        print("test_loss = {}".format(test_loss))

        return test_loss

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y


        return inputs

    def fit(self, num_epochs, optimizer, verbose=False):
        training_set_s = self.assemble_datasets()
        inp_train_s = next(iter(training_set_s))[0]
        self.approximate_solution = self.approximate_solution.to(device)
        training_set_s = inp_train_s.to(device)


        training_set_s.requires_grad = True
        history = list()

        test_mu_quake = torch.tensor([-0.132, 0.114])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            if self.test_on == 'True':
                test_input = self.get_test_loss_input(256,0.1,test_mu_quake)
                test_loss = self.compute_test_loss(test_input,test_mu_quake)
                wandb.log({"Test Loss": np.log10(test_loss.item())})
            if self.visualize == 'True':
                time_list = [0.0,0.05,0.1,0.5]
                for i in time_list:
                    plot_input = torch.clone(training_set_s)
                    plot_input[:, 0] = i
                    residual, U_with_ansatz, U_withouth_ansatz = self.get_solid_residual(plot_input)

                    residual_x = residual[0:int(len(residual) / 2)].detach().cpu().numpy()
                    residual_y = residual[int(len(residual) / 2):].detach().cpu().numpy()

                    U_with_x = U_with_ansatz[:, 0].detach().cpu().numpy()
                    U_with_y = U_with_ansatz[:, 1].detach().cpu().numpy()

                    U_without_x = U_withouth_ansatz[:, 0].detach().cpu().numpy()
                    U_without_y = U_withouth_ansatz[:, 1].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual ux @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual ux @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual uy @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual uy @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_y @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_y @ time = {} scatter".format(i): wandb.Image(fig)})
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(training_set_s)
                #if loss.requires_grad:
                loss.backward()
                history.append(loss.item())
                #del loss
                return loss
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

class Pinns_with_helper:

    def __init__(self, n_collocation_points,wandb_on,config):
        if config['Network']['activation'] == 'tanh':
             self.activation = nn.Tanh()

        else:
            print("unknown activation function", config['Network'].activation)
            exit()
        self.n_collocation_points = n_collocation_points
        self.wandb_on = wandb_on

        self.domain_extrema = torch.tensor([[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
                                                [float(config['domain']['xmin']), float(config['domain']['xmax'])], [float(config['domain']['ymin']), float(config['domain']['ymax'])],
                                                ])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=2,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3,activation=self.activation)
        if wandb_on:
            wandb.watch(self.approximate_solution, log_freq=100)


        if config['initial_condition']['source_function'] == 'explosion':
            self.source_function = initial_conditions.initial_condition_explosion
        elif config['initial_condition']['source_function'] == 'explosion_conditioned':
            self.source_function = initial_conditions.initial_condition_explosion_conditioned
        elif config['initial_condition']['source_function'] == 'explosion_two_sources':
            self.source_function = initial_conditions.initial_condition_explosion_two_sources
        elif config['initial_condition']['source_function'] == 'gaussian':
            self.source_function = initial_conditions.initial_condition_gaussian
        elif config['initial_condition']['source_function'] == 'donut':
            self.source_function = initial_conditions.initial_condition_donut
        else:
            print(config['initial_condition']['source_function'],'explosion', 'explosion' ==config['initial_condition']['source_function'])
            raise Exception("Source function {} is not implemented".format(config['initial_condition']['source_function']))




        self.t1 = float(config['initial_condition']['t1'])
        self.t1 = torch.tensor(self.t1)
        self.t1 = self.t1.to(device)
        self.sigma_quake = float(config['parameters']['sigma_quake'])
        self.sigma_quake = torch.tensor(self.sigma_quake)
        self.sigma_quake = self.sigma_quake.to(device)
        self.rho_solid = float(config['parameters']['rho_solid'])
        self.rho_solid = torch.tensor(self.rho_solid)
        self.rho_solid = self.rho_solid.to(device)
        self.parameter_model = config['parameters']['model_type']
        self.lambda_solid = config['parameters']['lambda_solid']
        self.mu_solid = config['parameters']['mu_solid']

        self.visualize = config['visualize']['visualize_on']
        self.test_on = config['test']['test_on']


    def pinn_model_eval(self, helper_output,input_tensor):
        # Evaluate the model with the given input tensor
        U_perturbation = self.approximate_solution(input_tensor)
        #t = input_tensor[:, 0]
        #u0x, u0y = initial_conditions.initial_condition_explosion_conditioned(input_tensor)
        u0x, u0y = self.source_function(input_tensor,self.sigma_quake)
        U = torch.zeros_like(U_perturbation)
        #t1 = float(config['initial_condition']['t1'])
        #print(U_perturbation[:, 0].shape,input_tensor[:, 0].shape,helper_output[:,0].shape)
        U[:, 0] = U_perturbation[:, 0] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0x * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2) + helper_output[:,0]
        U[:, 1] = U_perturbation[:, 1] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0y * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2) + helper_output[:,1]
        return U

    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
            #print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))

        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid ))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid ))
            #print(lambda_m.shape,mu_m.shape)
            #print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))

        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
            #print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))

        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        return input_s

    def assemble_datasets(self):
        input_s1 = self.add_solid_points()

        training_set_s1 = DataLoader(torch.utils.data.TensorDataset(input_s1),
                                     batch_size=int(self.n_collocation_points), shuffle=False, num_workers=4)
        return training_set_s1

    def compute_solid_loss(self, helper_output,input_s):

        U = self.pinn_model_eval(helper_output,input_s)

        #u_x = U[:, 0].unsqueeze(1)
        #u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(U[:, 0].unsqueeze(1).sum(), input_s, create_graph=True)[0]

        gradient_y = torch.autograd.grad(U[:, 1].unsqueeze(1).sum(), input_s, create_graph=True)[0]

        #dt_x = gradient_x[:, 0]
        #dx_x = gradient_x[:, 1]
        #dy_x = gradient_x[:, 2]
        #dt_y = gradient_y[:, 0]
        #dx_y = gradient_y[:, 1]
        #dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(gradient_x[:, 0].sum(), input_s, create_graph=True)[0][:, 0]

        dt2_y = torch.autograd.grad(gradient_y[:, 0].sum(), input_s, create_graph=True)[0][:, 0]

        #diag_1 = 2.0 * dx_x
        #diag_2 = 2.0 * dy_y
        #off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((2.0 * gradient_x[:, 1], gradient_x[:, 2] + gradient_y[:, 1])), torch.stack((gradient_x[:, 2] + gradient_y[:, 1], 2.0 * gradient_y[:, 2]))), dim=1)


        stress_tensor_00 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[0, 0]

        stress_tensor_off_diag = 2.0 * self.mu_m * eps[0, 1]

        stress_tensor_11 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[1, 1]

        #stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     #torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)
        off_diag_grad = torch.autograd.grad(stress_tensor_off_diag.sum(), input_s, create_graph=True)[0]


        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)

        div_stress[0, :] = torch.autograd.grad(stress_tensor_00.sum(), input_s, create_graph=True)[0][:, 1] + \
                           off_diag_grad[:, 2]

        div_stress[1, :] = off_diag_grad[:, 1] + \
                           torch.autograd.grad(stress_tensor_11.sum(), input_s, create_graph=True)[0][:, 2]


        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)

        residual_solid = self.rho_solid * dt2_combined - div_stress

        residual_solid = residual_solid.reshape(-1, )


        loss_solid = torch.mean(abs(residual_solid) ** 2)


        return loss_solid

    def get_solid_residual(self, helper_output,input_s):
        U = self.pinn_model_eval(helper_output,input_s)
        u_x = U[:, 0].unsqueeze(1)
        u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(u_x.sum(), input_s, create_graph=True)[0]
        gradient_y = torch.autograd.grad(u_y.sum(), input_s, create_graph=True)[0]
        dt_x = gradient_x[:, 0]
        dx_x = gradient_x[:, 1]
        dy_x = gradient_x[:, 2]
        dt_y = gradient_y[:, 0]
        dx_y = gradient_y[:, 1]
        dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(dt_x.sum(), input_s, create_graph=True)[0][:, 0]
        dt2_y = torch.autograd.grad(dt_y.sum(), input_s, create_graph=True)[0][:, 0]

        diag_1 = 2.0 * dx_x
        diag_2 = 2.0 * dy_y
        off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((diag_1, off_diag)), torch.stack((off_diag, diag_2))), dim=1)

        stress_tensor_00 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[0, 0]
        stress_tensor_off_diag = 2.0 * self.mu_m * eps[0, 1]
        stress_tensor_11 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[1, 1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        div_stress[0, :] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[1, :] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]

        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        residual_solid = self.rho_solid * dt2_combined - div_stress
        residual_solid = residual_solid.reshape(-1, )

        U_without_ansatz = self.approximate_solution(input_s)

        return residual_solid, U, U_without_ansatz

    def compute_loss(self, helper_output,inp_train_s):
        loss_solid = self.compute_solid_loss(helper_output,inp_train_s)
        loss = torch.log10(loss_solid)
        if self.wandb_on:
            wandb.log({"loss": loss.item()})
            wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        return loss

    def compute_test_loss(self, helper_network,test_input, mu_quake):

        # Number of points along x and y axis, total number of points will be numpoints_sqrt**2
        numpoints_sqrt = 256
        print("start")

        res_list_ux = []
        res_list_uy = []
        res_list_u = []
        time_list = np.linspace(0, 1, 101).tolist()
        for i in time_list:

            time = i
            inputs = test_input
            inputs[:, 0] = time
            inputs = inputs.to(device)
            helper_output = helper_network.pinn_model_eval(inputs)

            ux = self.pinn_model_eval(helper_output,inputs)[:, 0]
            uy = self.pinn_model_eval(helper_output,inputs)[:, 1]

            ux_out = ux.detach()
            uy_out = uy.detach()

            np_ux_out = ux_out.cpu().numpy()
            np_uy_out = uy_out.cpu().numpy()

            B_ux = np.reshape(np_ux_out, (-1, int(np.sqrt(np_ux_out.shape[0]))))
            B_uy = np.reshape(np_uy_out, (-1, int(np.sqrt(np_uy_out.shape[0]))))
            B = np.sqrt(B_uy ** 2 + B_ux ** 2)
            res_list_ux.append(B_ux)
            res_list_uy.append(B_uy)
            res_list_u.append(B)

        res_ux = np.dstack(res_list_ux)
        res_uy = np.dstack(res_list_uy)
        res_ux = np.rollaxis(res_ux, -1)
        res_uy = np.rollaxis(res_uy, -1)
        res_u = np.dstack(res_list_u)
        res_u = np.rollaxis(res_u, -1)
        s_ux = 5 * np.mean(np.abs(res_uy))
        s_uy = 5 * np.mean(np.abs(res_uy))
        s_u = 5 * np.mean(np.abs(res_u))


        f, axarr = plt.subplots(3, 3, figsize=(15, 20))
        plt.subplots_adjust(hspace=-0.1, wspace=0.1)
        test_loss = 0

        file_name_x = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_x.pkl'.format(self.parameter_model, mu_quake[0],mu_quake[1])
        file_name_y = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_y.pkl'.format(self.parameter_model, mu_quake[0],mu_quake[1])
        file_name_u = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_u.pkl'.format(self.parameter_model, mu_quake[0],mu_quake[1])

        with open(file_name_x, 'rb') as f_:
            res_list_devito_x = pickle.load(f_)
        with open(file_name_y, 'rb') as f_:
            res_list_devito_y = pickle.load(f_)
        with open(file_name_u, 'rb') as f_:
            res_list_devito_u = pickle.load(f_)

        s = 5 * np.mean((np.abs(res_list_devito_x[0])))



        for h in range(0, len(res_list_uy)):
            diffu = ((res_u[h, :, :]) - (res_list_devito_u[h]))
            diffu[0:9, :] = 0
            diffu[:, 0:9] = 0
            diffu[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
            diffu[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
            test_loss = test_loss + np.sqrt(np.mean(diffu**2))


            if h == 0 or h == int(len(res_list_uy) / 4) or h == int(len(res_list_uy) / 3) or h == int(
                    len(res_list_uy) / 2) or h == len(res_list_uy) - 2:
                diffx = ((res_uy[h, :, :]) - (res_list_devito_x[h]))
                diffx[0:9, :] = 0
                diffx[:, 0:9] = 0
                diffx[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffx[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1x = axarr[0][0].imshow(res_uy[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2x = axarr[0][1].imshow(res_list_devito_x[h], 'bwr', vmin=-s, vmax=s)
                im3x = axarr[0][2].imshow(diffx, 'bwr', vmin=-s, vmax=s)

                diffy = (res_ux[h, :, :]) - (res_list_devito_y[h])
                diffy[0:9, :] = 0
                diffy[:, 0:9] = 0
                diffy[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffy[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1y = axarr[1][0].imshow((res_ux[h, :, :]), 'bwr', vmin=-s, vmax=s)
                im2y = axarr[1][1].imshow((res_list_devito_y[h]), 'bwr', vmin=-s, vmax=s)
                im3y = axarr[1][2].imshow(diffy, 'bwr', vmin=-s, vmax=s)


                im1u = axarr[2][0].imshow(res_u[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2u = axarr[2][1].imshow(res_list_devito_u[h], 'bwr', vmin=-s, vmax=s)
                im3u = axarr[2][2].imshow(diffu, 'bwr', vmin=-s, vmax=s)


                axarr[0][0].set_title("PINN", fontsize=25, pad=20)
                axarr[0][1].set_title("Devito", fontsize=25, pad=20)
                axarr[0][2].set_title("Difference", fontsize=25, pad=20)
                f.subplots_adjust(right=0.8)
                cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                f.colorbar(im3x, cax=cbar_ax)

                if self.wandb_on:
                    wandb.log({"Test set difference @ time = {}".format(h): wandb.Image(f)})


        test_loss = test_loss / len(res_list_uy)
        #print("test_loss = {}".format(test_loss))

        return test_loss

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y


        return inputs

    def fit(self, helper_network,num_epochs, optimizer, verbose=False):
        training_set_s = self.assemble_datasets()
        inp_train_s = next(iter(training_set_s))[0]
        self.approximate_solution = self.approximate_solution.to(device)
        training_set_s = inp_train_s.to(device)
        helper_network.approximate_solution = helper_network.approximate_solution.to(device)
        helper_output = helper_network.pinn_model_eval(training_set_s)
        helper_output = helper_output.to(device)
        #print("training set shape = ",inp_train_s.shape)



        training_set_s.requires_grad = True
        history = list()

        test_mu_quake = torch.tensor([0.0, 0.0])
        test_mu_quake = test_mu_quake.to(device)
        test_input = self.get_test_loss_input(256, 0.1, test_mu_quake)
        # Loop over epochs
        for epoch in range(num_epochs):
            if self.test_on == 'True':

                test_loss = self.compute_test_loss(helper_network,test_input,test_mu_quake)
                wandb.log({"Test Loss": np.log10(test_loss.item())})
            if self.visualize == 'True':
                time_list = [0.0,0.05,0.1,0.5]
                for i in time_list:
                    plot_input = torch.clone(training_set_s)
                    plot_input[:, 0] = i
                    residual, U_with_ansatz, U_withouth_ansatz = self.get_solid_residual(helper_output,plot_input)

                    residual_x = residual[0:int(len(residual) / 2)].detach().cpu().numpy()
                    residual_y = residual[int(len(residual) / 2):].detach().cpu().numpy()

                    U_with_x = U_with_ansatz[:, 0].detach().cpu().numpy()
                    U_with_y = U_with_ansatz[:, 1].detach().cpu().numpy()

                    U_without_x = U_withouth_ansatz[:, 0].detach().cpu().numpy()
                    U_without_y = U_withouth_ansatz[:, 1].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual ux @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual ux @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual uy @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual uy @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_y @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_y @ time = {} scatter".format(i): wandb.Image(fig)})
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(helper_output,training_set_s)
                #if loss.requires_grad:
                loss.backward()
                history.append(loss.item())
                #del loss
                return loss
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

class Pinns_memory_recuced(Pinns):

    def __init__(self, n_collocation_points,wandb_on,config):
        if config['Network']['activation'] == 'tanh':
             self.activation = nn.Tanh()

        else:
            print("unknown activation function", config['Network'].activation)
            exit()
        self.n_collocation_points = n_collocation_points
        self.wandb_on = wandb_on

        self.domain_extrema = torch.tensor([[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
                                                [float(config['domain']['xmin']), float(config['domain']['xmax'])], [float(config['domain']['ymin']), float(config['domain']['ymax'])],
                                                ])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=2,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3,activation=self.activation)
        if wandb_on:
            wandb.watch(self.approximate_solution, log_freq=100)


        if config['initial_condition']['source_function'] == 'explosion':
            self.source_function = initial_conditions.initial_condition_explosion
        elif config['initial_condition']['source_function'] == 'explosion_conditioned':
            self.source_function = initial_conditions.initial_condition_explosion_conditioned
        elif config['initial_condition']['source_function'] == 'explosion_two_sources':
            self.source_function = initial_conditions.initial_condition_explosion_two_sources
        elif config['initial_condition']['source_function'] == 'gaussian':
            self.source_function = initial_conditions.initial_condition_gaussian
        elif config['initial_condition']['source_function'] == 'donut':
            self.source_function = initial_conditions.initial_condition_donut
        else:
            print(config['initial_condition']['source_function'],'explosion', 'explosion' ==config['initial_condition']['source_function'])
            raise Exception("Source function {} is not implemented".format(config['initial_condition']['source_function']))




        self.t1 = float(config['initial_condition']['t1'])
        self.t1 = torch.tensor(self.t1)
        self.t1 = self.t1.to(device)
        self.sigma_quake = float(config['parameters']['sigma_quake'])
        self.sigma_quake = torch.tensor(self.sigma_quake)
        self.sigma_quake = self.sigma_quake.to(device)
        self.rho_solid = float(config['parameters']['rho_solid'])
        self.rho_solid = torch.tensor(self.rho_solid)
        self.rho_solid = self.rho_solid.to(device)
        self.parameter_model = config['parameters']['model_type']
        self.lambda_solid = config['parameters']['lambda_solid']
        self.mu_solid = config['parameters']['mu_solid']

        self.visualize = config['visualize']['visualize_on']
        self.test_on = config['test']['test_on']


    def pinn_model_eval(self, input_tensor):
        # Evaluate the model with the given input tensor
        U_perturbation = self.approximate_solution(input_tensor)
        #t = input_tensor[:, 0]
        #u0x, u0y = initial_conditions.initial_condition_explosion_conditioned(input_tensor)
        u0x, u0y = self.source_function(input_tensor,self.sigma_quake)
        U = torch.zeros_like(U_perturbation)
        #t1 = float(config['initial_condition']['t1'])
        U[:, 0] = U_perturbation[:, 0] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0x * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2)
        U[:, 1] = U_perturbation[:, 1] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0y * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2)
        return U

    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
            #print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))

        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid ))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid ))
            #print(lambda_m.shape,mu_m.shape)
            #print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))

        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
            #print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))

        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        return input_s

    def assemble_datasets(self):
        input_s1 = self.add_solid_points()

        training_set_s1 = DataLoader(torch.utils.data.TensorDataset(input_s1),
                                     batch_size=int(self.n_collocation_points), shuffle=False, num_workers=4)
        return training_set_s1

    def compute_solid_loss(self, input_s):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("1: ",t,r,a)
        U = self.pinn_model_eval(input_s)
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("2: ", t, r, a)
        #u_x = U[:, 0].unsqueeze(1)
        #u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(U[:, 0].unsqueeze(1).sum(), input_s, create_graph=True)[0]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("3: ", t, r, a)
        gradient_y = torch.autograd.grad(U[:, 1].unsqueeze(1).sum(), input_s, create_graph=True)[0]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("4: ", t, r, a)
        #dt_x = gradient_x[:, 0]
        #dx_x = gradient_x[:, 1]
        #dy_x = gradient_x[:, 2]
        #dt_y = gradient_y[:, 0]
        #dx_y = gradient_y[:, 1]
        #dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(gradient_x[:, 0].sum(), input_s, create_graph=True)[0][:, 0]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("5: ", t, r, a)
        dt2_y = torch.autograd.grad(gradient_y[:, 0].sum(), input_s, create_graph=True)[0][:, 0]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("6: ", t, r, a)

        #diag_1 = 2.0 * dx_x
        #diag_2 = 2.0 * dy_y
        #off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((2.0 * gradient_x[:, 1], gradient_x[:, 2] + gradient_y[:, 1])), torch.stack((gradient_x[:, 2] + gradient_y[:, 1], 2.0 * gradient_y[:, 2]))), dim=1)
        del gradient_x
        del gradient_y
        torch.cuda.empty_cache()
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("7: ", t, r, a)
        stress_tensor_00 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[0, 0]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("8: ", t, r, a)
        stress_tensor_off_diag = 2.0 * self.mu_m * eps[0, 1]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("9: ", t, r, a)
        stress_tensor_11 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[1, 1]
        del eps
        torch.cuda.empty_cache()
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("10: ", t, r, a)
        #stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     #torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)
        off_diag_grad = torch.autograd.grad(stress_tensor_off_diag.sum(), input_s, create_graph=True)[0]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("11: ", t, r, a)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("12: ", t, r, a)
        div_stress[0, :] = torch.autograd.grad(stress_tensor_00.sum(), input_s, create_graph=True)[0][:, 1] + \
                           off_diag_grad[:, 2]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("13: ", t, r, a)
        div_stress[1, :] = off_diag_grad[:, 1] + \
                           torch.autograd.grad(stress_tensor_11.sum(), input_s, create_graph=True)[0][:, 2]
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("14: ", t, r, a)
        del stress_tensor_11
        del stress_tensor_00
        del stress_tensor_off_diag
        del off_diag_grad
        torch.cuda.empty_cache()
        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("15: ", t, r, a)
        residual_solid = self.rho_solid * dt2_combined - div_stress
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("16: ", t, r, a)
        del div_stress
        del dt2_combined
        torch.cuda.empty_cache()
        residual_solid = residual_solid.reshape(-1, )
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("17: ", t, r, a)

        loss_solid = torch.mean(abs(residual_solid) ** 2)
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("18: ", t, r, a)

        del residual_solid
        torch.cuda.empty_cache()
        print("del del")
        return loss_solid

    def get_solid_residual(self, input_s):
        U = self.pinn_model_eval(input_s)
        u_x = U[:, 0].unsqueeze(1)
        u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(u_x.sum(), input_s, create_graph=True)[0]
        gradient_y = torch.autograd.grad(u_y.sum(), input_s, create_graph=True)[0]
        dt_x = gradient_x[:, 0]
        dx_x = gradient_x[:, 1]
        dy_x = gradient_x[:, 2]
        dt_y = gradient_y[:, 0]
        dx_y = gradient_y[:, 1]
        dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(dt_x.sum(), input_s, create_graph=True)[0][:, 0]
        dt2_y = torch.autograd.grad(dt_y.sum(), input_s, create_graph=True)[0][:, 0]

        diag_1 = 2.0 * dx_x
        diag_2 = 2.0 * dy_y
        off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((diag_1, off_diag)), torch.stack((off_diag, diag_2))), dim=1)

        stress_tensor_00 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[0, 0]
        stress_tensor_off_diag = 2.0 * self.mu_m * eps[0, 1]
        stress_tensor_11 = self.lambda_m * (eps[0, 0] + eps[1, 1]) + 2.0 * self.mu_m * eps[1, 1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        div_stress[0, :] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[1, :] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]

        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        residual_solid = self.rho_solid * dt2_combined - div_stress
        residual_solid = residual_solid.reshape(-1, )

        U_without_ansatz = self.approximate_solution(input_s)

        return residual_solid, U, U_without_ansatz

    def compute_loss(self, inp_train_s):
        loss_solid = self.compute_solid_loss(inp_train_s)
        loss = torch.log10(loss_solid)
        if self.wandb_on:
            wandb.log({"loss": loss.item()})
            wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        return loss

    def compute_test_loss(self, test_input, mu_quake):

        # Number of points along x and y axis, total number of points will be numpoints_sqrt**2
        numpoints_sqrt = 256
        print("start")

        res_list_ux = []
        res_list_uy = []
        res_list_u = []
        time_list = np.linspace(0, 1, 101).tolist()
        for i in time_list:

            time = i
            inputs = test_input
            inputs[:, 0] = time
            inputs = inputs.to(device)

            ux = self.pinn_model_eval(inputs)[:, 0]
            uy = self.pinn_model_eval(inputs)[:, 1]
            ux_out = ux.detach()
            uy_out = uy.detach()

            np_ux_out = ux_out.cpu().numpy()
            np_uy_out = uy_out.cpu().numpy()

            B_ux = np.reshape(np_ux_out, (-1, int(np.sqrt(np_ux_out.shape[0]))))
            B_uy = np.reshape(np_uy_out, (-1, int(np.sqrt(np_uy_out.shape[0]))))
            B = np.sqrt(B_uy ** 2 + B_ux ** 2)
            res_list_ux.append(B_ux)
            res_list_uy.append(B_uy)
            res_list_u.append(B)

        res_ux = np.dstack(res_list_ux)
        res_uy = np.dstack(res_list_uy)
        res_ux = np.rollaxis(res_ux, -1)
        res_uy = np.rollaxis(res_uy, -1)
        res_u = np.dstack(res_list_u)
        res_u = np.rollaxis(res_u, -1)
        s_ux = 5 * np.mean(np.abs(res_uy))
        s_uy = 5 * np.mean(np.abs(res_uy))
        s_u = 5 * np.mean(np.abs(res_u))


        f, axarr = plt.subplots(3, 3, figsize=(15, 20))
        plt.subplots_adjust(hspace=-0.1, wspace=0.1)
        test_loss = 0

        file_name_x = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_x.pkl'.format(self.parameter_model, mu_quake[0],mu_quake[1])
        file_name_y = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_y.pkl'.format(self.parameter_model, mu_quake[0],mu_quake[1])
        file_name_u = 'pre_computed_test_devito/{}/mu=[{}, {}]/res_u.pkl'.format(self.parameter_model, mu_quake[0],mu_quake[1])

        with open(file_name_x, 'rb') as f_:
            res_list_devito_x = pickle.load(f_)
        with open(file_name_y, 'rb') as f_:
            res_list_devito_y = pickle.load(f_)
        with open(file_name_u, 'rb') as f_:
            res_list_devito_u = pickle.load(f_)

        s = 5 * np.mean((np.abs(res_list_devito_x[0])))



        for h in range(0, len(res_list_uy)):
            diffu = ((res_u[h, :, :]) - (res_list_devito_u[h]))
            diffu[0:9, :] = 0
            diffu[:, 0:9] = 0
            diffu[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
            diffu[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
            test_loss = test_loss + np.sqrt(np.mean(diffu**2))


            if h == 0 or h == int(len(res_list_uy) / 4) or h == int(len(res_list_uy) / 3) or h == int(
                    len(res_list_uy) / 2) or h == len(res_list_uy) - 2:
                diffx = ((res_uy[h, :, :]) - (res_list_devito_x[h]))
                diffx[0:9, :] = 0
                diffx[:, 0:9] = 0
                diffx[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffx[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1x = axarr[0][0].imshow(res_uy[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2x = axarr[0][1].imshow(res_list_devito_x[h], 'bwr', vmin=-s, vmax=s)
                im3x = axarr[0][2].imshow(diffx, 'bwr', vmin=-s, vmax=s)

                diffy = (res_ux[h, :, :]) - (res_list_devito_y[h])
                diffy[0:9, :] = 0
                diffy[:, 0:9] = 0
                diffy[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
                diffy[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0
                im1y = axarr[1][0].imshow((res_ux[h, :, :]), 'bwr', vmin=-s, vmax=s)
                im2y = axarr[1][1].imshow((res_list_devito_y[h]), 'bwr', vmin=-s, vmax=s)
                im3y = axarr[1][2].imshow(diffy, 'bwr', vmin=-s, vmax=s)


                im1u = axarr[2][0].imshow(res_u[h, :, :], 'bwr', vmin=-s, vmax=s)
                im2u = axarr[2][1].imshow(res_list_devito_u[h], 'bwr', vmin=-s, vmax=s)
                im3u = axarr[2][2].imshow(diffu, 'bwr', vmin=-s, vmax=s)


                axarr[0][0].set_title("PINN", fontsize=25, pad=20)
                axarr[0][1].set_title("Devito", fontsize=25, pad=20)
                axarr[0][2].set_title("Difference", fontsize=25, pad=20)
                f.subplots_adjust(right=0.8)
                cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                f.colorbar(im3x, cax=cbar_ax)

                if self.wandb_on:
                    wandb.log({"Test set difference @ time = {}".format(h): wandb.Image(f)})


        test_loss = test_loss / len(res_list_uy)
        print("test_loss = {}".format(test_loss))

        return test_loss

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y


        return inputs

    def fit(self, num_epochs, optimizer, verbose=False):
        training_set_s = self.assemble_datasets()
        inp_train_s = next(iter(training_set_s))[0]
        self.approximate_solution = self.approximate_solution.to(device)
        training_set_s = inp_train_s.to(device)


        training_set_s.requires_grad = True
        history = list()

        test_mu_quake = torch.tensor([0.0, 0.0])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            if self.test_on == 'True':
                test_input = self.get_test_loss_input(256,0.1,test_mu_quake)
                test_loss = self.compute_test_loss(test_input,test_mu_quake)
                wandb.log({"Test Loss": np.log10(test_loss.item())})
            if self.visualize == 'True':
                time_list = [0.0,0.05,0.1,0.5]
                for i in time_list:
                    plot_input = torch.clone(training_set_s)
                    plot_input[:, 0] = i
                    residual, U_with_ansatz, U_withouth_ansatz = self.get_solid_residual(plot_input)

                    residual_x = residual[0:int(len(residual) / 2)].detach().cpu().numpy()
                    residual_y = residual[int(len(residual) / 2):].detach().cpu().numpy()

                    U_with_x = U_with_ansatz[:, 0].detach().cpu().numpy()
                    U_with_y = U_with_ansatz[:, 1].detach().cpu().numpy()

                    U_without_x = U_withouth_ansatz[:, 0].detach().cpu().numpy()
                    U_without_y = U_withouth_ansatz[:, 1].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual ux @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual ux @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual uy @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual uy @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_y @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_y @ time = {} scatter".format(i): wandb.Image(fig)})
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(training_set_s)
                #if loss.requires_grad:
                loss.backward()
                history.append(loss.item())
                #del loss
                return loss
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history


class Global_NSources_Conditioned_Pinns(Pinns):
    def __init__(self, n_collocation_points,wandb_on,config):
        # Call the parent class's initializer
        super().__init__(n_collocation_points, wandb_on, config)

        # Modify the existing member variables
        self.domain_extrema = torch.tensor([
            [float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
            [float(config['domain']['xmin']), float(config['domain']['xmax'])],
            [float(config['domain']['ymin']), float(config['domain']['ymax'])],
            [-1.0, 1.0],
            [-1.0, 1.0]
        ])  # Space dimension

        self.approximate_solution = NeuralNet(
            input_dimension=5,
            output_dimension=2,
            n_hidden_layers=int(config['Network']['n_hidden_layers']),
            neurons=int(config['Network']['n_neurons']),
            regularization_param=0.,
            regularization_exp=2.,
            retrain_seed=3,
            activation=self.activation
        )


        self.n_sources = int(config["initial_condition"]["n_sources"])
        print(self.n_sources)

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))

        n_different_sources = int(self.n_sources)

        # Generate random source locations within the box of -0.5 to 0.5 for both x and y
        np.random.seed(42)  # To ensure repeatability
        source_x = np.random.uniform(-0.5, 0.5, n_different_sources)
        source_y = np.random.uniform(-0.5, 0.5, n_different_sources)

        # Repeat source locations for the corresponding collocation points
        source_idx = np.tile(np.arange(n_different_sources), int(self.n_collocation_points / n_different_sources))
        #print(input_s[:, 3].shape, torch.tensor(source_x[source_idx], dtype=torch.float32).shape)
        input_s[:, 3] = torch.tensor(source_x[source_idx], dtype=torch.float32)
        input_s[:, 4] = torch.tensor(source_y[source_idx], dtype=torch.float32)


        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
            #print(lambda_m.shape, sys.getsizeof((lambda_m)), sys.getsizeof(mu_m))
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid ))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid ))
            #print(lambda_m.shape,mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
            #print(lambda_m.shape,sys.getsizeof((lambda_m)),sys.getsizeof(mu_m))
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        return input_s

    def get_test_loss_input(self,numpoints_sqrt,time,mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]

        return inputs

class Relative_Distance_NSources_Conditioned_Pinns(Pinns):
    def __init__(self, n_collocation_points,wandb_on,config):
        super().__init__(n_collocation_points, wandb_on, config)


        self.domain_extrema = torch.tensor([[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
                                                [float(config['domain']['xmin']), float(config['domain']['xmax'])], [float(config['domain']['ymin']), float(config['domain']['ymax'])]
                                               ,[-1.0, 1.0],[-1.0, 1.0]
                                                ])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=5, output_dimension=2,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3,activation=self.activation)
        self.n_sources = int(config["initial_condition"]["n_sources"])
        #TODO: Hard coded
        self.source_function = initial_conditions.initial_condition_explosion_conditioned_relative

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))

        # Generate random source locations within the box of -0.5 to 0.5 for both x and y
        np.random.seed(42)  # To ensure repeatability
        source_x = np.random.uniform(-0.5, 0.5, self.n_sources)
        source_y = np.random.uniform(-0.5, 0.5, self.n_sources)

        # Repeat source locations for the corresponding collocation points
        source_idx = np.tile(np.arange(self.n_sources), int(self.n_collocation_points /self.n_sources))
        input_s[:, 1] = input_s[:, 1] - torch.tensor(source_x[source_idx], dtype=torch.float32)
        input_s[:, 2] = input_s[:, 2] -  torch.tensor(source_y[source_idx], dtype=torch.float32)

        #Lambda and mu should be computed with gloabl coordinates and not relative coordinates
        #Otherwise no matter the source location the parameter model will always be “centered" around (x,y) = (0,0)
        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid ))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid ))
            #print(lambda_m.shape,mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        input_s[:, 3] = torch.tensor(source_x[source_idx], dtype=torch.float32)
        input_s[:, 4] = torch.tensor(source_y[source_idx], dtype=torch.float32)
        #print(input_s)

        return input_s

    def get_test_loss_input(self,numpoints_sqrt,time,mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]
        inputs[:, 1] = inputs[:, 1] - inputs[:, 3]
        inputs[:, 2] = inputs[:, 2] - inputs[:, 4]

        return inputs

class Relative_Distance_FullDomain_Conditioned_Pinns(Pinns):
    def __init__(self, n_collocation_points,wandb_on,config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.domain_extrema = torch.tensor([[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
                                                [float(config['domain']['xmin']), float(config['domain']['xmax'])], [float(config['domain']['ymin']), float(config['domain']['ymax'])]
                                               ,[float(config['domain']['source_xmin']), float(config['domain']['source_xmax'])],[float(config['domain']['source_ymin']), float(config['domain']['source_ymax'])]
                                                ])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=5, output_dimension=2,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3,activation=self.activation)

        # TODO: Hard coded
        self.source_function = initial_conditions.initial_condition_explosion_conditioned_relative

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        # Lambda and mu should be computed with gloabl coordinates and not relative coordinates
        # Otherwise no matter the source location the parameter model will always be “centered" around (x,y) = (0,0)
        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid))
            #print(lambda_m.shape, mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)
        input_s[:, 1] = input_s[:, 1] - input_s[:,3]
        input_s[:, 2] = input_s[:, 2] - input_s[:,4]
        #print(input_s)

        return input_s

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]
        inputs[:, 1] = inputs[:, 1] - inputs[:, 3]
        inputs[:, 2] = inputs[:, 2] - inputs[:, 4]

        return inputs

class Global_NSources_Conditioned_Lame_Pinns(Pinns):
    def __init__(self, n_collocation_points, wandb_on,config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.domain_extrema = torch.tensor([[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
                                                [float(config['domain']['xmin']), float(config['domain']['xmax'])], [float(config['domain']['ymin']), float(config['domain']['ymax'])],
                                            [-1.0, 1.0], [-1.0, 1.0],[-1.0, 1.0],[-1.0, 1.0]])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=7, output_dimension=2,
                                              n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                              neurons=int(config['Network']['n_neurons']),
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=3,activation=self.activation)

        self.n_sources = int(config["initial_condition"]["n_sources"])

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))

        # Generate random source locations within the box of -0.5 to 0.5 for both x and y
        np.random.seed(42)  # To ensure repeatability
        source_x = np.random.uniform(-0.5, 0.5, self.n_sources)
        source_y = np.random.uniform(-0.5, 0.5, self.n_sources)

        # Repeat source locations for the corresponding collocation points
        source_idx = np.tile(np.arange(self.n_sources), int(self.n_collocation_points / self.n_sources))
        input_s[:, 1] = input_s[:, 1]
        input_s[:, 2] = input_s[:, 2]
        input_s[:, 3] = torch.tensor(source_x[source_idx], dtype=torch.float32)
        input_s[:, 4] = torch.tensor(source_y[source_idx], dtype=torch.float32)

        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid))
            #print(lambda_m.shape, mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        return input_s

    def compute_solid_loss(self, input_s):
        U = self.pinn_model_eval(input_s)
        u_x = U[:, 0].unsqueeze(1)
        u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(u_x.sum(), input_s, create_graph=True)[0]
        gradient_y = torch.autograd.grad(u_y.sum(), input_s, create_graph=True)[0]
        dt_x = gradient_x[:, 0]
        dx_x = gradient_x[:, 1]
        dy_x = gradient_x[:, 2]
        dt_y = gradient_y[:, 0]
        dx_y = gradient_y[:, 1]
        dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(dt_x.sum(), input_s, create_graph=True)[0][:, 0]
        dt2_y = torch.autograd.grad(dt_y.sum(), input_s, create_graph=True)[0][:, 0]

        diag_1 = 2.0 * dx_x
        diag_2 = 2.0 * dy_y
        off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((diag_1, off_diag)), torch.stack((off_diag, diag_2))), dim=1)

        stress_tensor_00 = input_s[:,5] * (eps[0, 0] + eps[1, 1]) + 2.0 * input_s[:,6] * eps[0, 0]
        stress_tensor_off_diag = 2.0 * input_s[:,6] * eps[0, 1]
        stress_tensor_11 = input_s[:,5] * (eps[0, 0] + eps[1, 1]) + 2.0 * input_s[:,6] * eps[1, 1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        div_stress[0, :] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[1, :] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]

        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        residual_solid = self.rho_solid * dt2_combined - div_stress
        residual_solid = residual_solid.reshape(-1, )

        loss_solid = torch.mean(abs(residual_solid) ** 2)

        return loss_solid

    def get_solid_residual(self, input_s):
        U = self.pinn_model_eval(input_s)
        u_x = U[:, 0].unsqueeze(1)
        u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(u_x.sum(), input_s, create_graph=True)[0]
        gradient_y = torch.autograd.grad(u_y.sum(), input_s, create_graph=True)[0]
        dt_x = gradient_x[:, 0]
        dx_x = gradient_x[:, 1]
        dy_x = gradient_x[:, 2]
        dt_y = gradient_y[:, 0]
        dx_y = gradient_y[:, 1]
        dy_y = gradient_y[:, 2]

        dt2_x = torch.autograd.grad(dt_x.sum(), input_s, create_graph=True)[0][:, 0]
        dt2_y = torch.autograd.grad(dt_y.sum(), input_s, create_graph=True)[0][:, 0]

        diag_1 = 2.0 * dx_x
        diag_2 = 2.0 * dy_y
        off_diag = dy_x + dx_y

        eps = 0.5 * torch.stack((torch.stack((diag_1, off_diag)), torch.stack((off_diag, diag_2))), dim=1)

        stress_tensor_00 = input_s[:, 5] * (eps[0, 0] + eps[1, 1]) + 2.0 * input_s[:, 6] * eps[0, 0]
        stress_tensor_off_diag = 2.0 * input_s[:, 6] * eps[0, 1]
        stress_tensor_11 = input_s[:, 5] * (eps[0, 0] + eps[1, 1]) + 2.0 * input_s[:, 6] * eps[1, 1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        div_stress[0, :] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[1, :] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]

        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        residual_solid = self.rho_solid * dt2_combined - div_stress
        residual_solid = residual_solid.reshape(-1, )

        U_without_ansatz = self.approximate_solution(input_s)

        return residual_solid, U, U_without_ansatz

    def compute_loss(self, inp_train_s):
        loss_solid = self.compute_solid_loss(inp_train_s)
        loss = torch.log10(loss_solid)
        if self.wandb_on:
            wandb.log({"loss": loss.item()})
            wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        return loss

    def fit(self, num_epochs, optimizer, verbose=False):
        training_set_s = self.assemble_datasets()
        inp_train_s = next(iter(training_set_s))[0]
        self.approximate_solution = self.approximate_solution.to(device)
        training_set_s = inp_train_s.to(device)
        training_set_s[:, 5] = self.lambda_m
        training_set_s[:, 6] = self.mu_m
        #print(training_set_s)
        training_set_s.requires_grad = True
        history = list()

        test_mu_quake = torch.tensor([-0.25, 0.25])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            if self.test_on == 'True':
                test_input = self.get_test_loss_input(256,0.1,test_mu_quake)
                test_loss = self.compute_test_loss(test_input,test_mu_quake)
                wandb.log({"Test Loss": np.log10(test_loss.item())})
            if self.visualize == 'True':
                time_list = [0.0,0.05,0.1,0.5]
                for i in time_list:
                    plot_input = torch.clone(training_set_s)
                    plot_input[:, 0] = i
                    residual, U_with_ansatz, U_withouth_ansatz = self.get_solid_residual(plot_input)

                    residual_x = residual[0:int(len(residual) / 2)].detach().cpu().numpy()
                    residual_y = residual[int(len(residual) / 2):].detach().cpu().numpy()

                    U_with_x = U_with_ansatz[:, 0].detach().cpu().numpy()
                    U_with_y = U_with_ansatz[:, 1].detach().cpu().numpy()

                    U_without_x = U_withouth_ansatz[:, 0].detach().cpu().numpy()
                    U_without_y = U_withouth_ansatz[:, 1].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual ux @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual ux @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual uy @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual uy @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_y @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_y @ time = {} scatter".format(i): wandb.Image(fig)})

            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(training_set_s)
                loss.backward()
                history.append(loss.item())
                return loss
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]

        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            lambda_m = mixture_model.compute_param(inputs[:, 1], inputs[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(inputs[:, 1], inputs[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid))
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(inputs[:,1], inputs[:,2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        inputs[:, 5] = lambda_m
        inputs[:, 6] = mu_m

        return inputs

class Global_FullDomain_Conditioned_Pinns(Pinns):
    def __init__(self, n_collocation_points, wandb_on, config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.domain_extrema = torch.tensor(
            [[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
             [float(config['domain']['xmin']), float(config['domain']['xmax'])],
             [float(config['domain']['ymin']), float(config['domain']['ymax'])]
                , [float(config['domain']['source_xmin']), float(config['domain']['source_xmax'])],
             [float(config['domain']['source_ymin']), float(config['domain']['source_ymax'])]
             ])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=5, output_dimension=2,
                                              n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                              neurons=int(config['Network']['n_neurons']),
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=3, activation=self.activation)

        self.n_epochs = int(config['optimizer']['n_epochs'])
        self.curriculum = config['Network']['curriculum']

    def assemble_datasets(self,epoch):
        input_s1 = self.add_solid_points(epoch)

        training_set_s1 = DataLoader(torch.utils.data.TensorDataset(input_s1),
                                     batch_size=int(self.n_collocation_points), shuffle=False, num_workers=4)
        return training_set_s1


    def add_solid_points(self,current_epoch):

        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        # Lambda and mu should be computed with gloabl coordinates and not relative coordinates
        # Otherwise no matter the source location the parameter model will always be “centered" around (x,y) = (0,0)

        #Curriculum

        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid))
            #print(lambda_m.shape, mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        if self.curriculum == 'True':
            min_time = self.domain_extrema[0, 0]
            original_min = self.domain_extrema[0, 0]
            original_max = self.domain_extrema[0, 1]

            max_time = ((current_epoch + 1) / self.n_epochs) * original_max

            # Rescale the time domain in-place
            input_s[:, 0] = min_time + (max_time - min_time) * (input_s[:, 0] - original_min) / (
                    original_max - original_min)
            print("min max time: ",min_time,max_time,torch.isnan(input_s).any(),torch.isinf(input_s).any())
            #input_s = input_s.to(device)
            #input_s.requires_grad = True

        return input_s

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]

        return inputs

    def fit(self, num_epochs, optimizer, verbose=False):
        if self.curriculum == 'False':
            training_set_s = self.assemble_datasets(0)
            inp_train_s = next(iter(training_set_s))[0]
            training_set_s = inp_train_s.to(device)
            training_set_s.requires_grad = True

        self.approximate_solution = self.approximate_solution.to(device)




        history = list()

        test_mu_quake = torch.tensor([-0.132, 0.114])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            if self.curriculum == 'True':
                training_set_s = self.assemble_datasets(epoch)
                inp_train_s = next(iter(training_set_s))[0]
                training_set_s = inp_train_s.to(device)
                training_set_s.requires_grad = True
                print(training_set_s)
            if self.test_on == 'True':
                test_input = self.get_test_loss_input(256,0.1,test_mu_quake)
                test_loss = self.compute_test_loss(test_input,test_mu_quake)
                wandb.log({"Test Loss": np.log10(test_loss.item())})
            if self.visualize == 'True':
                time_list = [0.0,0.05,0.1,0.5]
                for i in time_list:
                    plot_input = torch.clone(training_set_s)
                    plot_input[:, 0] = i
                    residual, U_with_ansatz, U_withouth_ansatz = self.get_solid_residual(plot_input)

                    residual_x = residual[0:int(len(residual) / 2)].detach().cpu().numpy()
                    residual_y = residual[int(len(residual) / 2):].detach().cpu().numpy()

                    U_with_x = U_with_ansatz[:, 0].detach().cpu().numpy()
                    U_with_y = U_with_ansatz[:, 1].detach().cpu().numpy()

                    U_without_x = U_withouth_ansatz[:, 0].detach().cpu().numpy()
                    U_without_y = U_withouth_ansatz[:, 1].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual ux @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual ux @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual uy @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual uy @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_y @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_y @ time = {} scatter".format(i): wandb.Image(fig)})
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(training_set_s)
                #if loss.requires_grad:
                loss.backward()
                history.append(loss.item())
                #del loss
                return loss
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

class Global_FullDomain_Conditioned_Pinns_Scramble_Resample(Pinns):
    def __init__(self, n_collocation_points, wandb_on, config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.domain_extrema = torch.tensor(
            [[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
             [float(config['domain']['xmin']), float(config['domain']['xmax'])],
             [float(config['domain']['ymin']), float(config['domain']['ymax'])]
                , [float(config['domain']['source_xmin']), float(config['domain']['source_xmax'])],
             [float(config['domain']['source_ymin']), float(config['domain']['source_ymax'])]
             ])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=5, output_dimension=2,
                                              n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                              neurons=int(config['Network']['n_neurons']),
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=3, activation=self.activation)

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0],scramble=True)

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        # Lambda and mu should be computed with gloabl coordinates and not relative coordinates
        # Otherwise no matter the source location the parameter model will always be “centered" around (x,y) = (0,0)
        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid))
            #print(lambda_m.shape, mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        return input_s

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]

        return inputs

    def fit(self, num_epochs, optimizer, verbose=False):
        training_set_s = self.assemble_datasets()
        inp_train_s = next(iter(training_set_s))[0]
        self.approximate_solution = self.approximate_solution.to(device)
        training_set_s = inp_train_s.to(device)

        training_set_s.requires_grad = True
        history = list()

        test_mu_quake = torch.tensor([-0.132, 0.114])
        test_mu_quake = test_mu_quake.to(device)

        # Loop over epochs
        for epoch in range(num_epochs):
            training_set_s = self.assemble_datasets()
            inp_train_s = next(iter(training_set_s))[0]
            training_set_s = inp_train_s.to(device)
            training_set_s.requires_grad = True

            if self.test_on == 'True':
                test_input = self.get_test_loss_input(256,0.1,test_mu_quake)
                test_loss = self.compute_test_loss(test_input,test_mu_quake)
                wandb.log({"Test Loss": np.log10(test_loss.item())})
            if self.visualize == 'True':
                time_list = [0.0,0.05,0.1,0.5]
                for i in time_list:
                    plot_input = torch.clone(training_set_s)
                    plot_input[:, 0] = i
                    residual, U_with_ansatz, U_withouth_ansatz = self.get_solid_residual(plot_input)

                    residual_x = residual[0:int(len(residual) / 2)].detach().cpu().numpy()
                    residual_y = residual[int(len(residual) / 2):].detach().cpu().numpy()

                    U_with_x = U_with_ansatz[:, 0].detach().cpu().numpy()
                    U_with_y = U_with_ansatz[:, 1].detach().cpu().numpy()

                    U_without_x = U_withouth_ansatz[:, 0].detach().cpu().numpy()
                    U_without_y = U_withouth_ansatz[:, 1].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual ux @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual ux @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=residual_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("residual uy @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"residual uy @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_with_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_with_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_with_y @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_x, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_x @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_x @ time = {} scatter".format(i): wandb.Image(fig)})

                    fig, ax = plt.subplots(figsize=(10, 6))
                    im_ux = ax.scatter(plot_input[:, 1].detach().cpu().numpy(), plot_input[:, 2].detach().cpu().numpy(),
                                       c=U_without_y, s=3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.plot(1, 1, marker="x")
                    ax.plot(-1, -1, marker="o")
                    ax.set_title("U_without_y @ time = {} scatter".format(i))
                    fig.colorbar(im_ux)
                    # plt.show()
                    if self.wandb_on:
                        wandb.log({"U_without_y @ time = {} scatter".format(i): wandb.Image(fig)})
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(training_set_s)
                loss.backward()
                history.append(loss.item())
                return loss
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

class Global_FullDomain_Conditioned_Pinns_FD_Ansatz(Pinns):
    def __init__(self, n_collocation_points, wandb_on, config):
        super().__init__(n_collocation_points, wandb_on, config)

        self.domain_extrema = torch.tensor(
            [[float(config['domain']['tmin']), float(config['domain']['tmax'])],  # Time dimension
             [float(config['domain']['xmin']), float(config['domain']['xmax'])],
             [float(config['domain']['ymin']), float(config['domain']['ymax'])]
                , [float(config['domain']['source_xmin']), float(config['domain']['source_xmax'])],
             [float(config['domain']['source_ymin']), float(config['domain']['source_ymax'])]
             ])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=5, output_dimension=2,
                                              n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                              neurons=int(config['Network']['n_neurons']),
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=3, activation=self.activation)

    def pinn_model_eval(self, input_tensor):
        # Evaluate the model with the given input tensor
        U_perturbation = self.approximate_solution(input_tensor)
        #t = input_tensor[:, 0]
        #u0x, u0y = initial_conditions.initial_condition_explosion_conditioned(input_tensor)
        u0x, u0y = self.source_function(input_tensor,self.sigma_quake)
        U = torch.zeros_like(U_perturbation)
        #t1 = float(config['initial_condition']['t1'])
        U[:, 0] = U_perturbation[:, 0] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0x * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2)
        U[:, 1] = U_perturbation[:, 1] * torch.tanh(2.5 *input_tensor[:, 0] / self.t1)**2 + u0y * torch.exp(-0.5 * (1.5 * input_tensor[:, 0]/self.t1)**2)
        return U

    def add_solid_points(self):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        input_s = self.convert(soboleng.draw(int(self.n_collocation_points)))
        # Lambda and mu should be computed with gloabl coordinates and not relative coordinates
        # Otherwise no matter the source location the parameter model will always be “centered" around (x,y) = (0,0)
        if self.parameter_model == 'mixture':
            mu_mixture = mixture_model.generate_mixture()
            lambda_mixture = mixture_model.generate_mixture()
            #mu_mixture = mu_mixture.to(device)
            #lambda_mixture = lambda_mixture.to(device)
            lambda_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], lambda_mixture)
            mu_m = mixture_model.compute_param(input_s[:, 1], input_s[:, 2], mu_mixture)
        elif self.parameter_model == 'constant':
            lambda_m = torch.full((self.n_collocation_points,), float(self.lambda_solid))
            mu_m = torch.full((self.n_collocation_points,), float(self.mu_solid))
            #print(lambda_m.shape, mu_m.shape)
        elif self.parameter_model == 'layered':
            lambda_m, mu_m = mixture_model.compute_lambda_mu_layers(input_s[:,1], input_s[:,2], 5)
        else:
            raise Exception("{} not implemented".format(self.parameter_model))
        self.lambda_m = lambda_m.to(device)
        self.mu_m = mu_m.to(device)

        return input_s

    def get_test_loss_input(self, numpoints_sqrt, time, mu_quake):
        soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake[0]
        inputs[:, 4] = mu_quake[1]

        return inputs





