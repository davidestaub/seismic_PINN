import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(2)
import torch.nn as nn
import torch
import os
import wandb
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


n_collocation_points = 80000
n_points_per_training_set = int(n_collocation_points)

lamda_solid = torch.tensor(2.0)#2.0 * 1e+8
lamda_solid = lamda_solid.to(device)
mu_solid = torch.tensor(3.0)#3.0 * 1e+8
mu_solid = mu_solid.to(device)
rho_solid = torch.tensor(1.0)#1000.0
rho_solid = rho_solid.to(device)
mu_quake = torch.tensor([0, -0.5])
mu_quake = mu_quake.to(device)
pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2)
pi = pi.to(device)


theta = torch.tensor(0.25 * torch.pi)
theta = theta.to(device)
# p wave speed
alpha = torch.sqrt((lamda_solid + 2.0 * mu_solid) / rho_solid)
alpha = alpha.to(device)
# s wave speed
beta = torch.sqrt(mu_solid / rho_solid)
beta=beta.to(device)

#plt.show()
T=torch.tensor(0.01)
T=T.to(device)
M0 = torch.tensor(0.5)
M0=M0.to(device)
mean2 = torch.zeros(2)
mean2[0] = mu_quake[0]
mean2[1] = mu_quake[1]
mean2 = mean2.to(device)


def radial_basis_function(x, mean, radius, time, decay_rate, rbf_sigma=0.1):
    # Calculate squared Euclidean distance of each point in x from the center.
    time = time+1
    squared_distance = ((x - mean) ** 2).sum(dim=1)
    radius = 0.4
    decay_rate = 10

    # RBF value at each point in x.
    rbf = torch.exp(-squared_distance*7)

    # Ball mask: 1.0 inside the ball, smoothly decaying to 0 outside.
    mask = torch.where(squared_distance <= radius ** 2, 1.0, rbf)


    # Calculate height based on time.
    height = torch.exp(-decay_rate * time).squeeze()

    # Result is height times mask.
    result = mask * height

    return result

def modified_radial_basis_function(x, mean, radius, time, decay_rate, rbf_sigma=0.1):
    # Calculate squared Euclidean distance of each point in x from the center.
    time = time + 1
    squared_distance = ((x - mean) ** 2).sum(dim=1)

    radius = 0.4
    decay_rate = 10

    # RBF value at each point in x.
    rbf = torch.exp(-squared_distance*7)

    # Ball mask: 1.0 inside the ball, smoothly decaying to 0 outside.
    mask = torch.where(squared_distance <= radius ** 2, 1.0, rbf)

    # Calculate height based on time.
    # Adjust decay to start only after time = 0.2
    height = torch.where(time <= 0.1, 0.9, torch.exp(-decay_rate * (time-0.1)).squeeze())

    # Result is height times mask.
    result = mask * height

    return result

def modified_radial_basis_function2(x, mean, radius, time, decay_rate, rbf_sigma=0.1):
    # Calculate squared Euclidean distance of each point in x from the center.
    time = time + 1
    squared_distance = ((x - mean) ** 2).sum(dim=1)

    radius = 0.4
    decay_rate = 5

    # RBF value at each point in x, modified for soft transition.
    rbf = torch.exp(-squared_distance*7 / (2 * (radius ** 2)))

    # Mask is now only RBF, providing a soft boundary.
    mask = rbf

    # Calculate height based on time.
    # Adjust decay to start only after time = 0.2
    height = torch.where(time <= 0.1, 0.9, torch.exp(-decay_rate * (time-0.1)).squeeze())

    height = torch.exp(-decay_rate * (time))

    # Result is height times mask.
    result = mask * height

    return result




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
    def __init__(self, n_collocation_points):
        self.n_collocation_points = n_collocation_points
        # Extrema of the solution domain (t,x(x,y)) in [0,0.1]x[-1,1]
        self.domain_extrema = torch.tensor([[-0.96, 0.0],  # Time dimension
                                            [-1.0, 1.0], [-1.0, 1.0]])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=2,
                                              n_hidden_layers=3,
                                              neurons=128,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=3)
        wandb.watch(self.approximate_solution, log_freq=100)
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        self.training_set_s = self.assemble_datasets()


    def pinn_model_eval(self,input_tensor):
        U_perturbation = self.approximate_solution(input_tensor)

        t = input_tensor[:, 0]
        x = input_tensor[:, 1]
        y = input_tensor[:, 2]

        r_abs = torch.sqrt(torch.pow((x - mu_quake[0]), 2) + torch.pow((y - mu_quake[1]), 2))
        r_abs = r_abs + 1e-50
        r_hat_x = (x - mu_quake[0]) / r_abs
        r_hat_y = (y - mu_quake[1]) / r_abs
        phi_hat_x = -1.0 * r_hat_y
        phi_hat_y = r_hat_x
        phi = torch.atan2(y - mu_quake[1], x - mu_quake[0])
        # mask = phi < 0
        # phi[mask] += 2.0 * torch.pi

        M0_dot_input1 = (t + 1 ) - r_abs / alpha
        M0_dot_input2 = (t + 1 ) - r_abs / beta

        M_dot1 = M0 / (T ** 2) * (M0_dot_input1 - 3.0 * T / 2.0) * torch.exp(
            -(M0_dot_input1 - 3.0 * T / 2.0) ** 2 / T ** 2)
        M_dot2 = M0 / (T ** 2) * (M0_dot_input2 - 3.0 * T / 2.0) * torch.exp(
            -(M0_dot_input2 - 3.0 * T / 2.0) ** 2 / T ** 2)

        A_FP_x = torch.sin(2.0 * theta) * torch.cos(phi) * r_hat_x
        A_FP_y = torch.sin(2.0 * theta) * torch.cos(phi) * r_hat_y

        A_FS_x = -torch.cos(theta) * torch.sin(phi) * phi_hat_x
        A_FS_y = -torch.cos(theta) * torch.sin(phi) * phi_hat_y

        far_field_x = (1.0 / (4.0 * torch.pi * alpha ** 3)) * A_FP_x * (1.0 / r_abs) * M_dot1 + (
                1.0 / (4.0 * torch.pi * beta ** 3)) * A_FS_x * (1.0 / r_abs) * M_dot2

        far_field_y = (1.0 / (4.0 * torch.pi * alpha ** 3)) * A_FP_y * (1.0 / r_abs) * M_dot1 + (
                1.0 / (4.0 * torch.pi * beta ** 3)) * A_FS_y * (1.0 / r_abs) * M_dot2


        analytic_x = far_field_x
        analytic_y = far_field_y

        U = torch.zeros_like(U_perturbation)

        xa = x.unsqueeze(-1)
        ya = y.unsqueeze(-1)
        point2 = torch.cat((xa, ya), dim=-1)

        analytic_activation_start2 = modified_radial_basis_function2(point2, mean2, 0, t, decay_rate=1, rbf_sigma=1.0)


        U[:, 0] = U_perturbation[:, 0] * (1.0 - analytic_activation_start2) + analytic_x * analytic_activation_start2
        U[:, 1] = U_perturbation[:, 1] * (1.0 - analytic_activation_start2) + analytic_y * analytic_activation_start2
        return U

    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]


    def add_solid_points(self):
        #ADDed sorting
        input_s = self.convert(self.soboleng.draw(int(self.n_collocation_points)))
        sorted_indices = torch.argsort(input_s[:, 0])
        input_s = input_s[sorted_indices]

        return input_s

    # Function returning the training sets as dataloader
    def assemble_datasets(self):
        input_s= self.add_solid_points()

       # input_init, output_init = self.add_init_points()
        training_set_s = DataLoader(torch.utils.data.TensorDataset(input_s),batch_size=self.n_collocation_points, shuffle=False, num_workers=8,pin_memory=True)

        #training_set_init = DataLoader(torch.utils.data.TensorDataset(input_init, output_init),batch_size=self.n_collocation_points, shuffle=False, num_workers=4)
        return training_set_s#, training_set_init

    def compute_solid_loss(self, input_s):
        U = self.pinn_model_eval(input_s)
        u_s = U[:, :2]  # Solid displacement components (x, y)
        u_x = u_s[:, 0].unsqueeze(1)
        u_y = u_s[:, 1].unsqueeze(1)
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
        # Reshape the gradients into tensors of shape [batch_size, 1]
        dx_x = dx_x.view(-1, 1)
        dy_x = dy_x.view(-1, 1)
        dx_y = dx_y.view(-1, 1)
        dy_y = dy_y.view(-1, 1)
        diag_1 = 2.0 * dx_x
        diag_2 = 2.0 * dy_y
        off_diag = dy_x + dx_y
        # Stack your tensors to a 2x2 tensor
        # The size of b will be (n_points, 2, 2)
        eps = 0.5 * torch.stack((torch.stack((diag_1, off_diag)), torch.stack((off_diag, diag_2))), dim=1)
        eps = eps.squeeze()

        # Double tensor contraction
        #stress_tensor = torch.einsum('ijkl,ijm->ikm', CU, eps)
        stress_tensor_00 = lamda_solid * (eps[0,0] + eps[1,1]) + 2.0 * mu_solid * eps[0,0]
        stress_tensor_off_diag = 2.0 * mu_solid * eps[0,1]
        #print("The off diagonal elements should be equal, and they are ? ",torch.eq(eps[0,1], eps[1,0]))
        stress_tensor_11 = lamda_solid * (eps[0,0] + eps[1,1]) + 2.0 * mu_solid * eps[1,1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)), torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(input_s.size(0), 2, dtype=torch.float32, device=input_s.device)
        div_stress[:, 0] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[:, 1] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]
        # Compute residuals
        dt2_combined = torch.stack((dt2_x, dt2_y), dim=1)
        residual_solid = rho_solid * dt2_combined - div_stress
        residual_solid = residual_solid.reshape(-1, )
        loss_solid = torch.mean(abs(residual_solid) **2)

        return loss_solid

    def compute_loss(self,inp_train_s):
        loss_solid = self.compute_solid_loss(inp_train_s)
        loss = torch.log10(loss_solid)
        wandb.log({"loss": loss.item()})
        wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        return loss

    def fit(self, num_epochs, optimizer, verbose=False):
        inp_train_s= next(iter(self.training_set_s))[0]
        self.approximate_solution = self.approximate_solution.to(device)
        #training_set_s = inp_train_s.to(device)
        #training_set_s.requires_grad = True
        history = list()
        # Loop over epochs
        for epoch in range(num_epochs):
            # Here we calculate the number of samples to include based on the current epoch
            #print(len(inp_train_s),(epoch + 1) * (int(self.n_collocation_points/num_epochs)))
            num_samples = min(len(inp_train_s), (epoch + 1) * (int(self.n_collocation_points/num_epochs)) + 2000)
            #print("num samples = ",num_samples)

            # We create a new DataLoader that only includes the first num_samples of the input_s
            training_set_s = inp_train_s[:num_samples]
            training_set_s = training_set_s.to(device)
            training_set_s.requires_grad = True
            #print("training_set_s",training_set_s)

            if verbose: print("################################ ", epoch, " ################################")
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(training_set_s)
                loss.backward()
                history.append(loss.item())
                return loss

            if epoch % 100 == 0:
                time_list = [-1.0,-0.9,-0.8,-0.7,-0.6,-0.5]
                res_list_ux = []
                res_list_uy = []
                numpoints_sqrt = 512
                inputs = self.soboleng.draw(int(pow(numpoints_sqrt, 2)))
                grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                                torch.linspace(-1.0, 1.0, numpoints_sqrt))
                grid_x = torch.reshape(grid_x, (-1,))
                grid_y = torch.reshape(grid_y, (-1,))

                for i in time_list:
                    inputs[:, 1] = grid_x
                    inputs[:, 2] = grid_y
                    inputs[:, 0] = i
                    inputs = inputs.to(device)
                    ux = self.pinn_model_eval(inputs)[:, 0]
                    uy = self.pinn_model_eval(inputs)[:, 1]
                    ux_out = ux.detach()
                    uy_out = uy.detach()
                    np_ux_out = ux_out.cpu().numpy()
                    np_uy_out = uy_out.cpu().numpy()
                    B_ux = np.reshape(np_ux_out, (-1, int(np.sqrt(np_ux_out.shape[0]))))
                    B_uy = np.reshape(np_uy_out, (-1, int(np.sqrt(np_uy_out.shape[0]))))
                    res_list_ux.append(B_ux)
                    res_list_uy.append(B_uy)

                res_ux = np.dstack(res_list_ux)
                res_uy = np.dstack(res_list_uy)
                res_ux = np.rollaxis(res_ux, -1)
                res_uy = np.rollaxis(res_uy, -1)

                for i in range(0, len(res_list_ux)):
                    plt.figure(figsize=(10, 6))
                    im_ux = plt.imshow(res_ux[i, :, :])
                    plt.xlabel("x ux")
                    plt.ylabel("y ux")
                    plt.title("ux @ time = {}".format(time_list[i]))
                    plt.colorbar(im_ux)
                    wandb.log({"ux @ time = {}".format(time_list[i]): plt})

                    plt.figure(figsize=(10, 6))
                    im_uy = plt.imshow(res_uy[i, :, :])
                    plt.xlabel("x uy")
                    plt.ylabel("y uy")
                    plt.title("uy @ time = {}".format(time_list[i]))
                    plt.colorbar(im_uy)
                    wandb.log({"uy @ time = {}".format(time_list[i]): plt})
                #self.redraw()
                #inp_train_s = next(iter(self.training_set_s))[0]
                #inp_train_s = inp_train_s.to(device)
                #inp_train_s.requires_grad = True
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        wandb.log({"final_loss": history[-1]})
        return history

wandb.init(project='Semester Thesis',name = 'solid only, analytical smoothed no timemarching epochs = 20000, m = 2, correct offset swish bit smoler start with 1000')
pinn = Pinns(n_points_per_training_set)
n_epochs = 20000
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                              lr=float(0.5),
                              max_iter=4,#200
                              max_eval=4, #200
                              history_size=2000, #2000
                              line_search_fn="strong_wolfe",
                              tolerance_grad=1e-8, #1e-8
                              tolerance_change=1.0 * np.finfo(float).eps)
hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_LBFGS,
                verbose=True)
torch.save(pinn.approximate_solution.state_dict(),
           "analytical_ansatz_march_swish_e20000_m2_Coffset.pth")