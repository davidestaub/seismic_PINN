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

n_collocation_points = 300
n_points_per_training_set = int(n_collocation_points/3)




# Constants:
g = torch.tensor([0.0, -9.81], dtype=torch.float32)

#example values lambda = 20GPa and mu = 30 GPa
lamda_solid = 20
mu_solid = 30
#values for T = 10c
#TODO construct as a function of temperature later for more realistic model
lamda_fluid = 1.426e-5
mu_fluid = 1.778e-5
#taken from internet density of granite = 1463.64kg/m3
rho_solid = 1463.64
#Initial values as provided by GPT4:
gamma = 1.4
rho0 = torch.full((n_points_per_training_set, 1), 1.225)
E0 = torch.zeros((n_points_per_training_set, 1))
v0x = torch.zeros((n_points_per_training_set, 1))
v0y = torch.zeros((n_points_per_training_set, 1))
p0 = torch.full((n_points_per_training_set, 1), 101325)
v_solid = torch.zeros((n_points_per_training_set, 1))
dv0x_dx = torch.zeros((n_points_per_training_set, 1))
dv0x_dy = torch.zeros((n_points_per_training_set, 1))
dv0y_dx = torch.zeros((n_points_per_training_set, 1))
dv0y_dy = torch.zeros((n_points_per_training_set, 1))
mu_quake = [2 / 2, 1 / 2]
sigma_quake = min(2, 1) * 0.05
#thermal condictivity for air with T = 10c
kappa = 0.02160

# constant temperature of 10c
t_10 = 283.15
#setting derivatives of T to zero for the case of constant temperature
#TODO change to more realistic model later
dT_dx =torch.full(v0x.shape, 0.0)
dT_dy =torch.full(v0x.shape, 0.0)

# Construct the 4th-order elastic tensor C^U
CU = torch.zeros(2, 2, 2, 2)
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                CU[i, j, k, l] = lamda_solid * (int(i == j) * int(k == l)) + mu_solid * (int(i == k) * int(j == l) + int(i == l) * int(j == k))


def compute_divergence(v_x, v_y, input_f):
    print("Tensor input_f requires_grad:", input_f.requires_grad)
    print("Tensor v_x requires_grad:", v_x.requires_grad)
    if v_x is v0x and v_y is v0y:
        dv_x_dx = dv0x_dx.squeeze()
        dv_y_dy = dv0y_dy.squeeze()
        return dv_x_dx + dv_y_dy
    elif not (v_x is v0x) and not(v_y is v0y):
        dv_x_dx = torch.autograd.grad(v_x.sum(), input_f, create_graph=True, retain_graph=True)[0][:, 1]
        dv_y_dy = torch.autograd.grad(v_y.sum(), input_f, create_graph=True, retain_graph=True)[0][:, 2]
        return dv_x_dx + dv_y_dy
    else:
        print("not comptatible values ")
        exit()

def compute_gradient(v_x, v_y, input_f):
    if v_x is v0x and v_y is v0y:
        dv_x_dx, dv_x_dy = dv0x_dx.squeeze(), dv0x_dy.squeeze()
        dv_y_dx, dv_y_dy = dv0y_dx.squeeze(), dv0y_dy.squeeze()
        return dv_x_dx, dv_x_dy, dv_y_dx, dv_y_dy
    elif not (v_x is v0x) and not (v_y is v0y):
        gradient_v_x = torch.autograd.grad(v_x.sum(), input_f, create_graph=True, retain_graph=True)[0]
        gradient_v_y = torch.autograd.grad(v_y.sum(), input_f, create_graph=True, retain_graph=True)[0]
        dv_x_dx, dv_x_dy = gradient_v_x[:, 1], gradient_v_x[:, 2]
        dv_y_dx, dv_y_dy = gradient_v_y[:, 1], gradient_v_y[:, 2]
        return dv_x_dx, dv_x_dy, dv_y_dx, dv_y_dy
    else:
        print("not comptatible values ")
        exit()


def Sigma_v(v_x, v_y, input_f, lambd, mu):
    div_v = compute_divergence(v_x, v_y, input_f)
    dv_x_dx, dv_x_dy, dv_y_dx, dv_y_dy = compute_gradient(v_x, v_y, input_f)

    print("div_v shape: ", div_v.shape)
    print("dv_x_dx shape: ", dv_x_dx.shape)
    print("dv_y_dx shape: ", dv_y_dx.shape)
    div_v = div_v.unsqueeze(1)  # Add dimension to match the shape of the stacked tensor

    #TODO: check with jakc I am really unsure about the dimensions here.
    sigma_v_x = lambd * div_v + 2 * mu * torch.stack([dv_x_dx, dv_y_dx], dim=1)
    sigma_v_y = lambd * div_v + 2 * mu * torch.stack([dv_x_dy, dv_y_dy], dim=1)

    return sigma_v_x, sigma_v_y

def Sigma_d1(U):
    return torch.zeros_like(U[:, 0]), torch.zeros_like(U[:, 0])

def Sigma_d2(v_prime_x, v_prime_y, input_f, lambd, mu):
    return Sigma_v(v_prime_x, v_prime_y, input_f, lambd, mu)

#TODO: I need to modify sigma_d3 if the input is v0x,v0y as these are predifenied values and note state variables so we cant take the derivative of them!!
def Sigma_d3(v_prime_x, v_prime_y, input_f, lambd, mu, kappa, dT_dx, dT_dy):
    Sigma_v_v0_x, Sigma_v_v0_y = Sigma_v(v0x, v0y, input_f, lambd, mu)
    Sigma_v_vprime_x, Sigma_v_vprime_y = Sigma_v(v_prime_x, v_prime_y, input_f, lambd, mu)
    print("shape Sigma_v_v0_x = ",Sigma_v_v0_x.shape)
    print("Sigma_v_vprime_x = ",Sigma_v_vprime_x.shape)
    print("v_prime_x",v_prime_x.shape)
    print("v0x", v0x.shape)
    print("dT_dx",dT_dx.shape)
    print("t1 ",Sigma_v_v0_x * v_prime_x)
    print("t2", Sigma_v_vprime_x * v0x)
    print("t3",kappa * dT_dx)
    Sigma_d3_x = Sigma_v_v0_x * v_prime_x + Sigma_v_vprime_x * v0x + kappa * dT_dx
    Sigma_d3_y = Sigma_v_v0_y * v_prime_y + Sigma_v_vprime_y * v0y + kappa * dT_dy

    return Sigma_d3_x, Sigma_d3_y

def get_p(rho,vx,vy,E):
    v_norm_squared = torch.norm(torch.stack((vx, vy), dim=1), dim=1) ** 2
    p = (gamma - 1) * (E - rho * v_norm_squared)
    return p

def get_p_prime(rho_prime,rho0_v_prime_x,rho0_v_prime_y,E_prime):
    vx_prime = rho0_v_prime_x / rho0
    vy_prime = rho0_v_prime_y / rho0
    return get_p(rho_prime +rho0,vx_prime+v0x,vy_prime+v0y,E_prime+E0) - p0

def Sigma_c1(U):
    rho_prime = U[:, 2]
    rho0_v_prime_x = U[:, 3]
    rho0_v_prime_y = U[:, 4]
    return rho0_v_prime_x + rho_prime * v0x, rho0_v_prime_y + rho_prime * v0y

def Sigma_c2(U, p_prime):
    rho0_v_prime_x = U[:, 3]
    rho0_v_prime_y = U[:, 4]
    v_prime_x = rho0_v_prime_x / rho0
    v_prime_y = rho0_v_prime_y / rho0

    sigma_c2_x = rho0 * v0x * v_prime_x + p_prime
    sigma_c2_y = rho0 * v0y * v_prime_y + p_prime

    return sigma_c2_x,sigma_c2_y

def Sigma_c3(U):
    rho_prime = U[:, 2]
    rho0_v_prime_x = U[:, 3]
    rho0_v_prime_y = U[:, 4]
    E_prime = U[:, 5]
    v_prime_x = rho0_v_prime_x / rho0
    v_prime_y = rho0_v_prime_y / rho0
    return (E0 + p0) * v_prime_x  + (E_prime + rho_prime) * v0x, (E0 + p0) * v_prime_y + (E_prime + rho_prime) * v0y

def G1(U):
    return 0, 0  # The first component of G is zero

#TODO: check
def G2(U, input_f, g):
    rho_prime = U[:, 2]
    rho0_v_prime_x = U[:, 3]
    rho0_v_prime_y = U[:, 4]


    term1 = rho0_v_prime_x + rho_prime * v0x
    term2 = rho0_v_prime_y + rho_prime * v0y

    grad_term1_x = torch.autograd.grad(term1.sum(), input_f, create_graph=True)[0][:, 1]
    grad_term2_y = torch.autograd.grad(term2.sum(), input_f, create_graph=True)[0][:, 2]

    dot_product = grad_term1_x * v0x + grad_term2_y * v0y

    return rho_prime * g[:, 0] - dot_product

def G3(U, g):
    rho_prime = U[:, 2]
    rho0_v_prime_x = U[:, 3]
    rho0_v_prime_y = U[:, 4]

    v0 = torch.stack([v0x, v0y], dim=1)
    rho0_v_prime = torch.stack([rho0_v_prime_x, rho0_v_prime_y], dim=1)
    rho_prime_v0 = rho_prime.unsqueeze(-1) * v0

    G3_result = (rho0_v_prime + rho_prime_v0 * g).sum(dim=-1)

    return G3_result

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
    def __init__(self, n_collocation_points):
        self.n_collocation_points = n_collocation_points

        # Extrema of the solution domain (t,x(x,y)) in [0,0.1]x[-1,1]
        self.domain_extrema = torch.tensor([[-1.0, 1.0],  # Time dimension
                                            [-1.0, 1.0], [-1.0, 1.0]])  # Space dimension

        # Number of space dimensions
        self.space_dimensions = 2

        # Parameter to balance role of data and PDE
        #TODO find suitable balance parameter, check if more than one neccessary
        self.lambda_u = 2

        # FF Dense NN to approximate the solution of the underlying heat equation
        #TODO find correct net architecture
        self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=6,
                                              n_hidden_layers=3,
                                              neurons=64,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)

        #wandb.watch(self.approximate_solution, log_freq=100)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

        # Training sets S_sb, S_tb, S_int as torch dataloader
        #TODO get correct training sets
        self.training_set_s, self.training_set_f, self.training_set_b = self.assemble_datasets()

    def pinn_model_eval(self, input_tensor, mu, sigma, solid_boundary=0.5, t0=-1.0):
        # Evaluate the model with the given input tensor
        U_perturbation = self.approximate_solution(input_tensor)

        # Apply initial conditions
        t = input_tensor[:, 0]

        x_part = torch.pow(input_tensor[:, 1] - mu[0], 2)
        y_part = torch.pow(input_tensor[:, 2] - mu[1], 2)
        exponent = -0.5 * torch.pow((torch.sqrt(x_part + y_part) / sigma), 2)
        earthquake_spike = torch.exp(exponent)

        solid_mask = input_tensor[:, 2] < solid_boundary
        u0x = earthquake_spike * solid_mask
        u0y = earthquake_spike * solid_mask

        #print("u0x = ",u0x)

        # Apply the initial conditions for each component of the state vector
        U = torch.zeros_like(U_perturbation)
        U[:, 0] = u0x + U_perturbation[:, 0] * torch.nn.functional.sigmoid(5 * (t / (-t0) - 1))
        U[:, 1] = u0y + U_perturbation[:, 1] * torch.nn.functional.sigmoid(5 * (t / (-t0) - 1))
        U[:, 2] = U_perturbation[:, 2] * torch.nn.functional.sigmoid(5 * (t / (-t0) - 1))
        U[:, 3] = U_perturbation[:, 3] * torch.nn.functional.sigmoid(5 * (t / (-t0) - 1))
        U[:, 4] = U_perturbation[:, 4] * torch.nn.functional.sigmoid(5 * (t / (-t0) - 1))
        U[:, 5] = U_perturbation[:, 5] * torch.nn.functional.sigmoid(5 * (t / (-t0) - 1))

        return U

    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    ################################################################################################


    def add_solid_points(self):
        # draw points from range -1 to 1
        input_s = self.convert(self.soboleng.draw(self.n_collocation_points))
        # Set the desired range for y values in the solid domain
        min_y = -1
        max_y = 0
        # Modify y values for the solid domain
        input_s[:, 2] = input_s[:, 2] * (max_y - min_y) / 2 + (max_y + min_y) / 2
        # Ensure that the maximum y value does not reach 0
        input_s[input_s[:, 2] == 0, 2] -= 1e-5

        # This is what you do when you dont care about the output
        output_s = torch.full((input_s.shape[0], 1), 0.0)
        #TODO check if I need to return the unused output

        return input_s,output_s


    def add_fluid_points(self):
        # draw points from range -1 to 1
        input_f = self.convert(self.soboleng.draw(self.n_collocation_points))
        # Set the desired range for y values in the solid domain
        min_y = 0
        max_y = 1
        # Modify y values for the fluid domain
        input_f[:, 2] = input_f[:, 2] * (max_y - min_y) / 2 + (max_y + min_y) / 2

        # Ensure that the minimum y value does not reach 0
        input_f[input_f[:, 2] == 0, 2] += 1e-5

        # This is what you do when you dont care about the output
        # TODO check if I need to return the unused output
        output_f = torch.full((input_f.shape[0], 1), 0.0)

        return input_f,output_f


    def add_boundary_points(self):
        # draw points from range -1 to 1
        input_b = self.convert(self.soboleng.draw(self.n_collocation_points))
        input_b[:, 2] = torch.full(input_b[:, 2].shape, 0.0)
        output_b = torch.full((input_b.shape[0], 1), 0.0)
        return input_b,output_b


    # Function returning the training sets as dataloader
    def assemble_datasets(self):
        input_s, output_s = self.add_solid_points()
        input_f, output_f = self.add_fluid_points()
        input_b, output_b = self.add_boundary_points()

        training_set_s = DataLoader(torch.utils.data.TensorDataset(input_s, output_s),
                                     batch_size=self.n_collocation_points, shuffle=False, num_workers=4)
        training_set_f = DataLoader(torch.utils.data.TensorDataset(input_f, output_f),
                                      batch_size=self.n_collocation_points, shuffle=False, num_workers=4)
        training_set_b = DataLoader(torch.utils.data.TensorDataset(input_b, output_b),
                                      batch_size=self.n_collocation_points, shuffle=False, num_workers=4)

        return training_set_s, training_set_f, training_set_b

    ################################################################################################


    def compute_solid_PDE_residual(self, input_s):
        U = self.pinn_model_eval(input_s, mu_quake, sigma_quake, solid_boundary=0.5, t0=-1.0)
        u_s = U[:, :2]  # Solid displacement components (x, y)
        print("us shape = ", u_s.shape)

        u_x = u_s[:, 0].unsqueeze(1)
        u_y = u_s[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(u_x.sum(), input_s, create_graph=True)[0]
        gradient_y = torch.autograd.grad(u_y.sum(), input_s, create_graph=True)[0]
        gradient = torch.cat((gradient_x, gradient_y), dim=1)
        print("gradient shape", gradient.shape)

        dt_x = gradient[:, 0]
        dx_x = gradient[:, 1]
        dy_x = gradient[:, 2]
        dt_y = gradient[:, 3]
        dx_y = gradient[:, 4]
        dy_y = gradient[:, 5]

        dt2_x = torch.autograd.grad(dt_x.sum(), input_s, create_graph=True)[0][:, 0]
        dt2_y = torch.autograd.grad(dt_y.sum(), input_s, create_graph=True)[0][:, 0]

        # Reshape the gradients into tensors of shape [batch_size, 1]
        dx_x = dx_x.view(-1, 1)
        dy_x = dy_x.view(-1, 1)
        dx_y = dx_y.view(-1, 1)
        dy_y = dy_y.view(-1, 1)

        # Combine the gradients into a single tensor
        gradient_us = torch.stack((torch.cat((dx_x, dx_y), dim=1), torch.cat((dy_x, dy_y), dim=1)), dim=1)

        # Compute the strain tensor
        print("gradient_us.shape:", gradient_us.shape)
        print("")
        strain_tensor = 0.5 * (gradient_us + torch.transpose(gradient_us, 1, 2))

        # Compute stress tensor courtesy of GPT-4
        print("strain tensor shape = ", strain_tensor.shape)
        CU_expanded = CU.expand(n_points_per_training_set, 2, 2, 2, 2)
        CU_expanded_permuted = CU_expanded.permute(0, 1, 3, 2, 4)

        temp = torch.einsum("abcde,abe->abcd", CU_expanded_permuted, strain_tensor)
        stress_tensor = torch.einsum("abcd,acd->abc", temp,
                                     strain_tensor)  # change 'ab' to 'abc' in the output subscript
        print("stress tensor shape", stress_tensor.shape)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(input_s.size(0), 2, dtype=torch.float32, device=input_s.device)

        print("div stress shape = ", div_stress.shape)
        print("input_s shape:", input_s.shape)
        div_stress[:, 0] = torch.autograd.grad(stress_tensor[:, 0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[:, 0, 1].sum(), input_s, create_graph=True)[0][:, 2]

        div_stress[:, 1] = torch.autograd.grad(stress_tensor[:, 1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[:, 1, 1].sum(), input_s, create_graph=True)[0][:, 2]

        # Compute residuals
        print("dt2 shape = ", dt2_x.shape, "dt2_y shape = ", dt2_y.shape, "div stress shape = ", div_stress.shape)
        dt2_combined = torch.stack((dt2_x, dt2_y), dim=1)
        residual_solid = rho_solid * dt2_combined - div_stress

        return residual_solid.reshape(-1, )

    def compute_fluid_PDE_residual(self,input_f):

        U = self.pinn_model_eval(input_f, mu_quake,sigma_quake, solid_boundary = 0.5, t0 = -1.0)
        rho_prime = U[:, 2]  # density perurbation
        rho0_v_prime_x = U[:,3] # rest density * velocity pertubation in x direction
        rho0_v_prime_y = U[:, 4]  #rest density * velocity pertubation in y direction
        E_prime = U[:,5] #Energy pertubation
        u = U[:,2:6] #combined
        p_prime = get_p_prime(rho_prime,rho0_v_prime_x,rho0_v_prime_y,E_prime)

        gradient_u = torch.autograd.grad(u.sum(), input_f, create_graph=True)[0]
        du_dt = gradient_u[:, 0]


        v_prime_x = rho0_v_prime_x / rho0.squeeze(1)
        print("shape of rho0_v_prime_x= ", rho0_v_prime_x.shape)
        print("shape of rho0= ", rho0.squeeze(1).shape)
        print("shape of v_prime_x= ", v_prime_x.shape)
        v_prime_y = rho0_v_prime_y / rho0.squeeze(1)



        # Compute Sigma_c
        Sigma_c_x = Sigma_c1(U)[0] + Sigma_c2(U, p_prime)[0] + Sigma_c3(U)[0]
        Sigma_c_y = Sigma_c1(U)[1] + Sigma_c2(U, p_prime)[1] + Sigma_c3(U)[1]

        # Compute the divergence

        div_Sigma_c_x = torch.autograd.grad(Sigma_c_x.sum(), input_f, create_graph=True)[0]
        div_Sigma_c_y = torch.autograd.grad(Sigma_c_y.sum(), input_f, create_graph=True)[0]
        div_Sigma_c_x = div_Sigma_c_x[:, 1]
        div_Sigma_c_y = div_Sigma_c_y[:, 2]

        div_Sigma_c = div_Sigma_c_x + div_Sigma_c_y


        Sigma_d1_x, Sigma_d1_y = Sigma_d1(U)
        #Sigma_d2_x, Sigma_d2_y = Sigma_d2(v_prime_x, v_prime_y, dx, dy, lamda_fluid, mu_fluid)
        #Sigma_d3_x, Sigma_d3_y = Sigma_d3(v_prime_x, v_prime_y, v0x, v0y, dx, dy, lamda_fluid, mu_fluid, dT_dx, dT_dy,
                                    #      kappa)

        Sigma_d_x = torch.stack([Sigma_d1_x, Sigma_d2(v_prime_x, v_prime_y, input_f, lamda_fluid, mu_fluid)[0],
                                 Sigma_d3(v_prime_x, v_prime_y, input_f, lamda_fluid, mu_fluid, kappa, dT_dx, dT_dy)[
                                     0]], dim=1)
        Sigma_d_y = torch.stack([Sigma_d1_y, Sigma_d2(v_prime_x, v_prime_y, input_f, lamda_fluid, mu_fluid)[1],
                                 Sigma_d3(v_prime_x, v_prime_y, input_f, lamda_fluid, mu_fluid, kappa, dT_dx, dT_dy)[
                                     1]], dim=1)

        #Sigma_d_x = torch.stack([Sigma_d1_x, Sigma_d2(dvprime_dx, dvprime_dy, lamda_fluid, mu_fluid)[:, 0], Sigma_d3(dvprime_dx, dvprime_dy,dv0_dx, dv0_dy,v_prime_x,v_prime_y, lamda_fluid, mu_fluid, dT_dx, dT_dy, kappa)[:, 0]], dim=1)
        #Sigma_d_y = torch.stack([Sigma_d1_y, Sigma_d2(dvprime_dx, dvprime_dy, lamda_fluid, mu_fluid)[:, 1], Sigma_d3(dvprime_dx, dvprime_dy,dv0_dx, dv0_dy,v_prime_x,v_prime_y, lamda_fluid, mu_fluid, dT_dx, dT_dy, kappa)[:, 1]], dim=1)

        div_Sigma_d_x = torch.autograd.grad(Sigma_d_x.sum(), input_f, create_graph=True)[0][:, 1]
        div_Sigma_d_y = torch.autograd.grad(Sigma_d_y.sum(), input_f, create_graph=True)[0][:, 2]

        div_Sigma_d = div_Sigma_d_x + div_Sigma_d_y

        G = torch.stack([G1(U), G2(U, input_f,g), G3(U, g)])

        residual = du_dt + div_Sigma_c - div_Sigma_d - G

        return residual.reshape(-1, )

    #TODO complete
    def compute_boundary_loss(self,input_b):
        n = torch.tensor([0.0, 1.0], device=input_b.device).unsqueeze(1)
        t = torch.tensor([1.0, 0.0], device=input_b.device).unsqueeze(1)
        U = self.pinn_model_eval(input_b, mu_quake,sigma_quake, solid_boundary = 0.5, t0 = -1.0)
        rho_prime = U[:, 2]  # density perturbation
        rho0_v_prime_x = U[:,3]
        rho0_v_prime_y = U[:,4]

        # 1. ∂nρf' = 0
        # Calculate the gradient of rho_prime
        grad_rho_prime = torch.autograd.grad(rho_prime.sum(), input_b, create_graph=True)[0][:, 1:]
        r1 = (grad_rho_prime @ n.unsqueeze(1)).squeeze()  # Compute the dot product

        # 2. vf.n - v+.n = 0
        v_prime_x = rho0_v_prime_x / rho0
        v_prime_y = rho0_v_prime_y / rho0
        v_f_x = v0x + v_prime_x
        v_f_y = v0y + v_prime_y
        v_f = torch.stack([v_f_x, v_f_y], dim=1)
        vf_dot_n = (v_f * n.T).sum(dim=1)
        v_plus_dot_n = (v_solid * n.T).sum(dim=1)
        r2 = vf_dot_n - v_plus_dot_n

        # 3. ∂n(vf * t )  = 0
        #can avoid computing again!
        vf_dot_t = (v_f * t.T).sum(dim=1)
        grad_vf_dot_t = torch.autograd.grad(vf_dot_t.sum(), input_b, create_graph=True)[0][:, 1:]
        r3 = (grad_vf_dot_t @ n.unsqueeze(1)).squeeze()  # Compute the dot product

        # 4. ρf' - ρ+' = 0
        r4 = rho_prime - rho_solid

        # 5. ∂n(grad Tf') = 0
        #TODO: this is only valid if assuming constant temperature in the atomsphere for more complex models this doe snot hold!!
        r5 = torch.zeros_like(r1)

        # 6. ∂n Xf' = 0, X = Sigmav(vf')
        v_prime = torch.stack([v_prime_x, v_prime_y], dim=1)
        gradient_v_prime = torch.autograd.grad(v_prime.sum(), input_b, create_graph=True)[0]
        dvprime_dx = gradient_v_prime[:, 1]
        dvprime_dy = gradient_v_prime[:, 2]

        Sigma_v_computed = Sigma_v(dvprime_dx, dvprime_dy, lamda_fluid, mu_fluid)

        Xf_prime = Sigma_v_computed
        # Compute the gradient of Xf_prime with respect to x and y only
        gradient_Xf_prime = torch.stack([
            torch.autograd.grad(Xf_prime[:, i, j].sum(), input_b, create_graph=True)[0][:, 1:]
            for i in range(2) for j in range(2)], dim=1).view(-1, 2, 2, 2)

        # Compute the dot product between the gradient of Xf_prime and the normal vector n
        dn_Xf_prime = (gradient_Xf_prime @ n.unsqueeze(-1)).squeeze(-1)
        r6 = dn_Xf_prime

        # Sigma_s n -  Sigma_f n = 0
        Sigma_v_reshaped = Sigma_v_computed.view(-1, 2, 2)
        n_flipped = -n
        normal_stress_fluid = torch.einsum('ijk,ik->ij', Sigma_v_reshaped, n_flipped)

        #### SOLID PART ###
        u_s = U[:, :2]  # Solid displacement components (x, y)
        gradient = torch.autograd.grad(u_s.sum(), input_b, create_graph=True)[0]
        dx = gradient[:, 1]
        dy = gradient[:, 2]
        # Combine the gradients into a single tensor
        gradient_us = torch.stack((dx, dy), dim=1)
        # Compute the strain tensor
        strain_tensor = 0.5 * (gradient_us + torch.transpose(gradient_us, 1, 2))
        # Compute stress tensor
        stress_tensor = torch.einsum("ijkl,ijl->ijk", CU, strain_tensor)
        normal_stress_solid = torch.einsum('ijk,ik->ij', stress_tensor, n_flipped)

        #### END OF SOLID PART #####
        r7 = normal_stress_solid - normal_stress_fluid

        # All CBCs:
        loss_r1 = torch.mean(torch.abs(r1) ** 2)
        loss_r2 = torch.mean(torch.abs(r2) ** 2)
        loss_r3 = torch.mean(torch.abs(r3) ** 2)
        loss_r4 = torch.mean(torch.abs(r4) ** 2)
        loss_r5 = torch.mean(torch.abs(r5) ** 2)
        loss_r6 = torch.mean(torch.abs(r6) ** 2)
        loss_r7 = torch.mean(torch.abs(r7) ** 2)

        boundary_loss = loss_r1 + loss_r2 + loss_r3 + loss_r4 + loss_r5 + loss_r6 + loss_r7

        return boundary_loss




    def compute_loss(self,inp_train_s, u_train_s, inp_train_f, u_train_f, inp_train_b,u_train_b,verbose):

        loss_solid = torch.mean(abs(self.compute_solid_PDE_residual(inp_train_s)) **2)
        loss_fluid = torch.mean(abs(self.compute_fluid_PDE_residual(inp_train_f)) ** 2)
        loss_boundary = self.compute_boundary_loss(inp_train_b)

        loss = torch.log10(loss_solid + loss_fluid + loss_boundary)

        #wandb.log({"loss": loss.item()})
        #wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        #wandb.log({"Fluid loss": torch.log10(loss_fluid).item()})
        #wandb.log({"Boundary loss": torch.log10(loss_boundary).item()})

        if verbose: print("Total loss: ", round(loss.item(), 4), "| Solid Loss: ", round(torch.log10(loss_solid).item(), 4),"| Fluid Loss: ", round(torch.log10(loss_fluid).item(), 4),"| Boundary Loss: ", round(torch.log10(loss_boundary).item(), 4))

        return loss






    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=False):

        inp_train_s, u_train_s = next(iter(self.training_set_s))
        inp_train_f, u_train_f = next(iter(self.training_set_f))
        inp_train_b, u_train_b = next(iter(self.training_set_b))

        inp_train_s = inp_train_s.to(device)
        u_train_s = u_train_s.to(device)
        inp_train_f = inp_train_f.to(device)
        u_train_f = u_train_f.to(device)
        inp_train_b = inp_train_b.to(device)

        #print(inp_train_s,inp_train_f)

        self.approximate_solution = self.approximate_solution.to(device)


        print("ON GPU?", inp_train_s.is_cuda)
        print("ON GPU?", inp_train_f.is_cuda)
        print("ON GPU?", u_train_s.is_cuda)
        print("ON GPU?", u_train_f.is_cuda)
        print("ON GPU?", inp_train_b.is_cuda)
        print("ON GPU?", u_train_b.is_cuda)
        print("ON GPU?", next(self.approximate_solution.parameters()).is_cuda)

        #TODO check which ones require grad
        inp_train_s.requires_grad = True
        inp_train_f.requires_grad = True
        inp_train_b.requires_grad = True

        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            def closure():

                optimizer.zero_grad()


                loss = self.compute_loss(inp_train_s, u_train_s, inp_train_f, u_train_f, inp_train_b,u_train_b,verbose=verbose)
                loss.backward()

                history.append(loss.item())
                return loss

            optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])

        return history

    ################################################################################################

#wandb.init(project='Semester Thesis',name = 'first test')

pinn = Pinns(n_points_per_training_set)

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
           "../pre_trained_models/first_test.pth")

