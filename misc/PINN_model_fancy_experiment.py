import configparser
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(2)
import torch.nn as nn
import torch
import os
import wandb
import initial_conditions

torch.set_printoptions(threshold=10000)
torch.set_printoptions(linewidth=200)

config = configparser.ConfigParser()
config.read("config.ini")

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Parameters
lamda_solid = float(config['parameters']['lambda_solid'])
mu_solid = float(config['parameters']['mu_solid'])
rho_solid = float(config['parameters']['rho_solid'])
c2 = float(config['parameters']['c2'])
mu_quake_x = float(config['parameters']['mu_quake_x'])
mu_quake_y = float(config['parameters']['mu_quake_y'])
mu_quake = [mu_quake_x, mu_quake_y]
mu_quake = torch.tensor(mu_quake)
mu_quake = mu_quake.to(device)
sigma_quake = float(config['parameters']['sigma_quake'])
radius = float(config['parameters']['radius'])
T = float(config['parameters']['T'])
T = torch.tensor(T)
M0 = float(config['parameters']['M0'])
M0 = torch.tensor(M0)
T = T.to(device)
M0 = M0.to(device)
a = float(config['initial_condition']['a'])
b = float(config['initial_condition']['b'])

pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2)
pi = pi.to(device)
theta = torch.tensor(0.25 * torch.pi)
theta.to(device)

# p wave speed
alpha = torch.tensor(np.sqrt((lamda_solid + 2.0 * mu_solid) / rho_solid))
# s wave speed
beta = torch.tensor(np.sqrt(mu_solid / rho_solid))

alpha = alpha.to(device)
beta = beta.to(device)


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
        if config['Network']['activation'] == 'tanh':
            self.activation = nn.Tanh()
        else:
            print("unknown activation function", config['Network'].activation)
            exit()
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
    def __init__(self, n_collocation_points):
        self.n_collocation_points = n_collocation_points
        self.domain_extrema = torch.tensor([[0.0, 1.0],  # Time dimension
                                            [-1.0, 1.0], [-1.0, 1.0],
                                            [-1.0,1.0],[-1.0,1.0]])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=5, output_dimension=2,
                                              n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                              neurons=int(config['Network']['n_neurons']),
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=3)
        wandb.watch(self.approximate_solution, log_freq=100)
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        self.training_set_s= self.assemble_datasets()

    def pinn_model_eval(self, input_tensor):
        # Evaluate the model with the given input tensor
        U_perturbation = self.approximate_solution(input_tensor)
        t = input_tensor[:, 0]
        u0x, u0y = initial_conditions.initial_condition_explosion_conditioned(input_tensor)
        U = torch.zeros_like(U_perturbation)
        t1 = float(config['initial_condition']['t1'])
        U[:, 0] = U_perturbation[:, 0] * torch.tanh(2.5 *t / t1)**2 + u0x * torch.exp(-0.5 * (1.5 * t/t1)**2)
        U[:, 1] = U_perturbation[:, 1] * torch.tanh(2.5 *t / t1)**2 + u0y * torch.exp(-0.5 * (1.5 * t/t1)**2)
        return U

    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    import numpy as np

    def add_solid_points(self):
        input_s = self.convert(self.soboleng.draw(int(self.n_collocation_points)))
        n_different_sources = 10

        # Generate random source locations within the box of -0.5 to 0.5 for both x and y
        np.random.seed(42)  # To ensure repeatability
        source_x = np.random.uniform(-0.5, 0.5, n_different_sources)
        source_y = np.random.uniform(-0.5, 0.5, n_different_sources)

        # Repeat source locations for the corresponding collocation points
        source_idx = np.tile(np.arange(n_different_sources), int(self.n_collocation_points / n_different_sources))
        input_s[:, 3] = torch.tensor(source_x[source_idx], dtype=torch.float32)
        input_s[:, 4] = torch.tensor(source_y[source_idx], dtype=torch.float32)
        print(input_s)

        return input_s


    # Function returning the training sets as dataloader
    def assemble_datasets(self):
        input_s1 = self.add_solid_points()

        training_set_s1 = DataLoader(torch.utils.data.TensorDataset(input_s1),
                                     batch_size=int(self.n_collocation_points), shuffle=False, num_workers=4)

        return training_set_s1


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

        stress_tensor_00 = lamda_solid * (eps[0, 0] + eps[1, 1]) + 2.0 * mu_solid * eps[0, 0]
        stress_tensor_off_diag = 2.0 * mu_solid * eps[0, 1]
        stress_tensor_11 = lamda_solid * (eps[0, 0] + eps[1, 1]) + 2.0 * mu_solid * eps[1, 1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        div_stress[0, :] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[1, :] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]

        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        residual_solid = rho_solid * dt2_combined - div_stress
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

        stress_tensor_00 = lamda_solid * (eps[0, 0] + eps[1, 1]) + 2.0 * mu_solid * eps[0, 0]
        stress_tensor_off_diag = 2.0 * mu_solid * eps[0, 1]
        stress_tensor_11 = lamda_solid * (eps[0, 0] + eps[1, 1]) + 2.0 * mu_solid * eps[1, 1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(2, input_s.size(0), dtype=torch.float32, device=input_s.device)
        div_stress[0, :] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[1, :] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]

        dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
        residual_solid = rho_solid * dt2_combined - div_stress
        residual_solid = residual_solid.reshape(-1, )

        U_without_ansatz = self.approximate_solution(input_s)

        return residual_solid, U, U_without_ansatz

    def compute_loss(self, inp_train_s):
        loss_solid = self.compute_solid_loss(inp_train_s)
        loss = torch.log10(loss_solid )
        wandb.log({"loss": loss.item()})
        wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        return loss

    def fit(self, num_epochs, optimizer, verbose=False):

        inp_train_s = next(iter(self.training_set_s))[0]
        self.approximate_solution = self.approximate_solution.to(device)
        training_set_s = inp_train_s.to(device)
        training_set_s.requires_grad = True
        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if config['visualize']['visualize_on'] == 'True':
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