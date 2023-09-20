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

torch.set_printoptions(threshold=10000)
torch.set_printoptions(linewidth=200)


torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


n_collocation_points = 90000
n_points_per_training_set = int(n_collocation_points)

#example values lambda = 20GPa and mu = 30 GPa
#lamda_solid = 20
lamda_solid = 2.0
mu_solid = 3.0
rho_solid = 1.0
#rho_fluid = 997
rho_fluid = 1.0
#c2 = 1450**2
c2 = 1.0
mu_quake = torch.tensor([0, 0])
mu_quake = mu_quake.to(device)
sigma_quake = min(2, 1) * 0.12
radius = 0.2
pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2)
pi = pi.to(device)
theta = torch.tensor(0.25 * torch.pi)
theta.to(device)
# p wave speed
alpha = torch.tensor(np.sqrt((lamda_solid + 2.0 * mu_solid) / rho_solid))
# s wave speed

beta = torch.tensor(np.sqrt(mu_solid / rho_solid))
T=torch.tensor(0.02)
M0 = torch.tensor(0.1)
alpha = alpha.to(device)
beta =beta.to(device)
T=T.to(device)
M0=M0.to(device)


def analytic_initial(input_tensor):

    t_initial = torch.full(input_tensor[:,0].shape, -0.9)
    t_initial = t_initial.to(device)
    print(t_initial.shape)
    x = input_tensor[: ,1]
    y = input_tensor[: ,2]

    r_abs = torch.sqrt(torch.pow((x - mu_quake[0]) ,2) + torch.pow((y - mu_quake[1]) ,2)+ 1e-5)
    r_abs = r_abs + 1e-5
    r_hat_x = (x - mu_quake[0] ) /r_abs
    r_hat_y = (y - mu_quake[1] ) /r_abs
    phi_hat_x = -1.0 * r_hat_y
    phi_hat_y = r_hat_x
    phi = torch.atan2(y-mu_quake[1],(x - mu_quake[0])+ 1e-5)

    print(t_initial.shape,r_abs.shape,alpha.shape)

    M0_dot_input1 = (t_initial + 1) - r_abs / alpha
    M0_dot_input2 = (t_initial + 1) - r_abs / beta


    M_dot1 = M0/(T**2) * (M0_dot_input1 - 3.0*T/2.0) * torch.exp(-(M0_dot_input1- 3.0*T/2.0)**2/T**2)
    M_dot2 = M0/(T**2) * (M0_dot_input2- 3.0*T/2.0) * torch.exp(-(M0_dot_input2- 3.0*T/2.0)**2/T**2)


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

    return analytic_x,analytic_y

def initial_condition_explosion(input_tensor, sigma=0.1):
    x = input_tensor[:, 1] - mu_quake[0]
    y = input_tensor[:, 2] - mu_quake[1]

    # Generate 2D Gaussian distribution
    gauss = torch.exp(- (x**2 + y**2) / (2*sigma**2))

    # Compute gradients of the Gaussian
    grad_x = -2 * x * gauss / (2*sigma**2)
    grad_y = -2 * y * gauss / (2*sigma**2)

    # Normalize gradients to get initial velocity field
    u0x = -grad_x / torch.max(torch.abs(grad_x))
    u0y = -grad_y / torch.max(torch.abs(grad_y))

    return u0x, u0y

def initial_condition_donut(input_tensor, inner_radius=0.1, transition_width=0.05):
    x = input_tensor[:, 1] - mu_quake[0]
    y = input_tensor[:, 2] - mu_quake[1]

    r = torch.sqrt((x ** 2 + y ** 2) + 1e-8)  # radius from the center of the quake
    theta = torch.atan2(y, x + 1e-8)  # angle from the positive x-axis

    # create a mask that is 1 inside the annulus and 0 elsewhere, with smooth transitions
    inside_outer_circle = 1 / (1 + torch.exp((r - radius) / transition_width))  # smoothly transition from 1 to 0 as r goes from radius to radius + transition_width
    outside_inner_circle = 1 / (1 + torch.exp((inner_radius - r) / transition_width))  # smoothly transition from 0 to 1 as r goes from inner_radius - transition_width to inner_radius
    inside_annulus = inside_outer_circle * outside_inner_circle

    # Rotate the vectors 90 degrees counterclockwise
    # In polar coordinates (r, theta), this corresponds to adding pi/2 to the angle.
    theta += np.pi / 2

    # Now compute the x and y coordinates of the vector in the rotated frame.
    # After rotation, the vector that was initially along the x-axis is now along the y-axis, and the vector that was initially along the y-axis is now along the negative x-axis.
    u0x = inside_annulus * r / radius * torch.sin(theta)
    u0y = -inside_annulus * r / radius * torch.cos(theta)

    return u0x, u0y

def initial_condition_gaussian(input_tensor):
    x_part = torch.pow(input_tensor[:, 1] - mu_quake[0], 2)
    y_part = torch.pow(input_tensor[:, 2] - mu_quake[1], 2)

    exponent = -0.5 * torch.pow((torch.sqrt(x_part + y_part + 1e-8) / sigma_quake), 2)
    earthquake_spike = torch.exp(exponent)
    u0x = earthquake_spike  # * solid_mask
    u0y = earthquake_spike  # * solid_mask

    return u0x,u0y


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
        # Extrema of the solution domain (t,x(x,y)) in [0,0.1]x[-1,1]
        self.domain_extrema = torch.tensor([[-1.0, 0.0],  # Time dimension
                                            [-1.0, 1.0], [-1.0, 1.0]])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=2,
                                              n_hidden_layers=3,
                                              neurons=256,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=3)
        wandb.watch(self.approximate_solution, log_freq=100)
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        #self.training_set_s1, self.training_set_s2 = self.assemble_datasets()
        self.training_set_s, self.training_set_s_t0 = self.assemble_datasets()
        #print(type(self.training_set_s))

    def pinn_model_eval(self, input_tensor, mu, sigma, solid_boundary=0.0, t0=-1.0):
        # Evaluate the model with the given input tensor
        U_perturbation = self.approximate_solution(input_tensor)
        # Apply initial conditions
        t = input_tensor[:, 0]
        u0x,u0y = analytic_initial(input_tensor)
        U = torch.zeros_like(U_perturbation)
        # sigmoid(5 * (2- ((x+1.25)*7)))
        # tanh(tanh(((x+1)*5)))
        U[:, 0] = U_perturbation[:, 0] * torch.tanh(torch.tanh((t + 1.0) * 5)) + u0x * torch.sigmoid(
            15.0 * (2.0 - ((t + 2.75) * 1.0)))
        #torch.tanh(torch.tanh((t + 1.0) * a)) for a in {2,20}
        #torch.sigmoid(15.0 * (b - ((t + 2.75) * 1.0))) for b in {1.8,2.2}
        U[:, 1] = U_perturbation[:, 1] * torch.tanh(torch.tanh((t + 1.0) * 5)) + u0y * torch.sigmoid(
            15.0 * (2.0 - ((t + 2.75) * 1.0)))
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

    def add_solid_points_t0(self):
        input_s = self.convert(self.soboleng.draw(int(self.n_collocation_points/10)))
        input_s[:, 0] = -1.0
        return input_s

    # Function returning the training sets as dataloader
    def assemble_datasets(self):
        input_s1= self.add_solid_points()
        input_st0 = self.add_solid_points_t0()
        #input_s2 = self.add_solid_points2()
       # input_init, output_init = self.add_init_points()
        training_set_s1 = DataLoader(torch.utils.data.TensorDataset(input_s1),batch_size=int(self.n_collocation_points), shuffle=False, num_workers=4)
        training_set_s_t0 = DataLoader(torch.utils.data.TensorDataset(input_st0),batch_size=int(self.n_collocation_points/10), shuffle=False, num_workers=4)

        #print(type(training_set_s))
        #training_set_init = DataLoader(torch.utils.data.TensorDataset(input_init, output_init),batch_size=self.n_collocation_points, shuffle=False, num_workers=4)
        return training_set_s1,training_set_s_t0#, training_set_init

    def compute_solid_loss(self, input_s):
        U = self.pinn_model_eval(input_s, mu_quake, sigma_quake, solid_boundary=0.0, t0=-1.0)
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

    def compute_no_init_velocity_loss(self,input_st0):
        U = self.pinn_model_eval(input_st0, mu_quake, sigma_quake, solid_boundary=0.0, t0=-1.0)
        u_s = U[:, :2]  # Solid displacement components (x, y)
        u_x = u_s[:, 0].unsqueeze(1)
        u_y = u_s[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(u_x.sum(), input_st0, create_graph=True)[0]
        gradient_y = torch.autograd.grad(u_y.sum(), input_st0, create_graph=True)[0]
        dt_x = gradient_x[:, 0]
        dt_y = gradient_y[:, 0]

        loss_no_init_velocity_loss = torch.mean(abs(dt_x)**2) + torch.mean(abs(dt_y)**2)
        return loss_no_init_velocity_loss


    def compute_loss(self,inp_train_s,inp_train_s_t0):
        loss_solid = self.compute_solid_loss(inp_train_s)
        loss_no_init_velocity = self.compute_no_init_velocity_loss(inp_train_s_t0)
        loss = torch.log10(loss_solid + loss_no_init_velocity)
        wandb.log({"loss": loss.item()})
        wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        wandb.log({"no init vel loss": torch.log10(loss_no_init_velocity).item()})
        return loss

    def fit(self, num_epochs, optimizer, verbose=False):

        inp_train_s= next(iter(self.training_set_s))[0]
        training_set_no_init_vel = next(iter(self.training_set_s_t0))[0]
        training_set_no_init_vel = training_set_no_init_vel.to(device)
        training_set_no_init_vel.requires_grad = True
        self.approximate_solution = self.approximate_solution.to(device)
        training_set_s = inp_train_s.to(device)
        training_set_s = training_set_s.to(device)
        training_set_s.requires_grad = True


        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):

            if verbose: print("################################ ", epoch, " ################################")
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(training_set_s,training_set_no_init_vel)
                loss.backward()
                history.append(loss.item())
                return loss

            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

wandb.init(project='Semester Thesis',name = 'analytical_explosion long run')
pinn = Pinns(n_points_per_training_set)
n_epochs = 200
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                              lr=float(1.2),
                              max_iter=300,
                              max_eval=300,
                              history_size=1000,
                              line_search_fn="strong_wolfe",
                              tolerance_grad=1e-8,
                              tolerance_change=1.0 * np.finfo(float).eps)

hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_LBFGS,
                verbose=True)
torch.save(pinn.approximate_solution.state_dict(),
           "analytical_explosion.pth")
