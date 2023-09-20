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
import wandb

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)

n_collocation_points = 30000
n_points_per_training_set = int(n_collocation_points)


#rho_fluid = 997
rho_fluid = 1.0
#c2 = 1450**2
c2 = 1.0
mu_quake = [0, 0.0]
sigma_quake = min(2, 1) * 0.12


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
        self.domain_extrema = torch.tensor([[-1.0, 0.0],  # Time dimension
                                            [-1.0, 1.0], [-1.0, 1.0]])  # Space dimension
        self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=2,
                                              n_hidden_layers=5,
                                              neurons=256,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)
        wandb.watch(self.approximate_solution, log_freq=100)
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        #self.training_set_f= self.assemble_datasets()
        self.training_set_f, self.training_set_f_t0 = self.assemble_datasets()

    def pinn_model_eval(self, input_tensor, mu, sigma, solid_boundary=0.0, t0=-1.0):
        # Evaluate the model with the given input tensor
        U_perturbation = self.approximate_solution(input_tensor)
        # Apply initial conditions
        t = input_tensor[:, 0]
        x_part = torch.pow(input_tensor[:, 1] - mu[0], 2)
        y_part = torch.pow(input_tensor[:, 2] - mu[1], 2)
        exponent = -0.5 * torch.pow((torch.sqrt(x_part + y_part + 1e-8) / sigma), 2)
        earthquake_spike = torch.exp(exponent)
        # solid_mask = input_tensor[:, 2] < solid_boundary
        u0x = earthquake_spike  # * solid_mask
        u0y = earthquake_spike  # * solid_mask
        U = torch.zeros_like(U_perturbation)
        # sigmoid(5 * (2- ((x+1.25)*7)))
        # tanh(tanh(((x+1)*5)))
        U[:, 0] = U_perturbation[:, 0] * torch.tanh(torch.tanh((t + 1.0) * 5)) + u0x * torch.sigmoid(
            15.0 * (2.0 - ((t + 2.75) * 1.0)))
        U[:, 1] = U_perturbation[:, 1] * torch.tanh(torch.tanh((t + 1.0) * 5)) + u0y * torch.sigmoid(
            15.0 * (2.0 - ((t + 2.75) * 1.0)))
        return U

    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]


    def add_fluid_points(self):
        # draw points from range -1 to 1
        input_f = self.convert(self.soboleng.draw(self.n_collocation_points))
        # Set the desired range for y values in the solid domain
       # min_y = 0
        #max_y = 1
        # Modify y values for the fluid domain
        #input_f[:, 2] = input_f[:, 2] * (max_y - min_y) / 2 + (max_y + min_y) / 2
        # Ensure that the minimum y value does not reach 0
        #input_f[input_f[:, 2] == 0, 2] += 1e-8

        return input_f

    def add_fluid_points_t0(self):
        input_f = self.convert(self.soboleng.draw(int(self.n_collocation_points/10)))
        input_f[:, 0] = -1.0
        return input_f




    # Function returning the training sets as dataloader
    def assemble_datasets(self):
        input_f = self.add_fluid_points()
        input_ft0 = self.add_fluid_points_t0()
        training_set_f = DataLoader(torch.utils.data.TensorDataset(input_f),batch_size=self.n_collocation_points, shuffle=False, num_workers=4)
        training_set_f_t0 = DataLoader(torch.utils.data.TensorDataset(input_ft0),batch_size=int(self.n_collocation_points / 10), shuffle=False, num_workers=4)

        return training_set_f,training_set_f_t0




    def redraw(self):
        self.training_set_s = self.assemble_datasets()

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


    def compute_fluid_loss(self,input_f):
        U = self.pinn_model_eval(input_f, mu_quake,sigma_quake, solid_boundary = 0.0, t0 = -1.0)
        u_x = U[:,0]
        u_y = U[:,1]
        gradient_ux = torch.autograd.grad(u_x.sum(), input_f, create_graph=True)[0]
        dux_dt = gradient_ux[:, 0]
        dux_dx = gradient_ux[:, 1]
        dux_dy = gradient_ux[:, 2]
        gradient_uy = torch.autograd.grad(u_y.sum(), input_f, create_graph=True)[0]
        duy_dt = gradient_uy[:, 0]
        duy_dx = gradient_uy[:, 1]
        duy_dy = gradient_uy[:, 2]
        dux_dt2 = torch.autograd.grad(dux_dt.sum(), input_f, create_graph=True)[0][:,0]
        dux_dx2 = torch.autograd.grad(dux_dx.sum(), input_f, create_graph=True)[0][:,1]
        dux_dy2 = torch.autograd.grad(dux_dy.sum(), input_f, create_graph=True)[0][:,2]
        duy_dt2 = torch.autograd.grad(duy_dt.sum(), input_f, create_graph=True)[0][:,0]
        duy_dx2 = torch.autograd.grad(duy_dx.sum(), input_f, create_graph=True)[0][:,1]
        duy_dy2 = torch.autograd.grad(duy_dy.sum(), input_f, create_graph=True)[0][:,2]
        x_residual = dux_dt2 - c2 * (dux_dx2 + dux_dy2)
        y_residual = duy_dt2 - c2 * (duy_dx2 + duy_dy2)
        l_x = torch.mean(abs(x_residual.reshape(-1, )) ** 2)
        l_y = torch.mean(abs(y_residual.reshape(-1, )) ** 2)

        return l_x + l_y

    def compute_loss(self, inp_train_f,inp_train_f_t0,verbose):
        loss_fluid = self.compute_fluid_loss(inp_train_f)
        loss_no_init_velocity = self.compute_no_init_velocity_loss(inp_train_f_t0)
        loss = torch.log10(loss_fluid +loss_no_init_velocity)
        wandb.log({"loss": loss.item()})
        wandb.log({"Fluid loss": torch.log10(loss_fluid).item()})
        #if verbose: print("Total loss: ", round(loss.item(), 4), "| Solid Loss: ", round(torch.log10(loss_solid).item(), 4),"| Fluid Loss: ", round(torch.log10(loss_fluid).item(), 4),"| Boundary Loss: ", round(torch.log10(loss_boundary).item(), 4))
        return loss

    def fit(self, num_epochs, optimizer, verbose=False):
        training_set_no_init_vel = next(iter(self.training_set_f_t0))[0]
        training_set_no_init_vel = training_set_no_init_vel.to(device)
        training_set_no_init_vel.requires_grad = True
        inp_train_f= next(iter(self.training_set_f))[0]
        inp_train_f = inp_train_f.to(device)
        print(inp_train_f)
        self.approximate_solution = self.approximate_solution.to(device)
        inp_train_f.requires_grad = True
        history = list()
        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")
            if epoch % 50 == 0:
                time_list = [-1.0,0.0,1.0]
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

                    ux = self.pinn_model_eval(inputs, mu_quake, sigma_quake, solid_boundary=0.0, t0=-1.0)[:, 0]
                    uy = self.pinn_model_eval(inputs, mu_quake, sigma_quake, solid_boundary=0.0, t0=-1.0)[:, 1]
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
                #inp_train_f = next(iter(self.training_set_f))[0]
                #inp_train_f = inp_train_f.to(device)
                #inp_train_f.requires_grad = True
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(inp_train_f,training_set_no_init_vel,verbose=verbose)
                loss.backward()
                history.append(loss.item())
                return loss
            optimizer.step(closure=closure)
        print('Final Loss:', history[-1])
        return history



wandb.init(project='Semester Thesis',name = 'center FLUID ONLY changed ANSATZ larger sigma like SEMINAR')
pinn = Pinns(n_points_per_training_set)
n_epochs = 100
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                              lr=float(0.3),
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
           "fluid_only-like_seminar.pth")

