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


n_collocation_points = 1000
n_points_per_training_set = int(n_collocation_points)

#example values lambda = 20GPa and mu = 30 GPa
#lamda_solid = 20
lamda_solid = 20.0
#mu_solid = 30
mu_solid = 30.0
#taken from internet density of granite = 1463.64kg/m3
#rho_solid = 1463.64
rho_solid = 100.0
#rho_fluid = 997
rho_fluid = 1.0
#c2 = 1450**2
c2 = 1.0
mu_quake = [0, -0.5]
sigma_quake = min(2, 1) * 0.05


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
        self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=2,
                                              n_hidden_layers=4,
                                              neurons=128,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=3)
        wandb.watch(self.approximate_solution, log_freq=100)
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        self.training_set_s1, self.training_set_s2 = self.assemble_datasets()
        #print(type(self.training_set_s))

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
            5.0 * (2.0 - ((t + 1.25) * 7.0)))
        U[:, 1] = U_perturbation[:, 1] * torch.tanh(torch.tanh((t + 1.0) * 5)) + u0y * torch.sigmoid(
            5.0 * (2.0 - ((t + 1.25) * 7.0)))
        return U

    #def redraw(self):
      #  self.training_set_s = self.assemble_datasets()

    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def convert2(self, tens):
       # assert (tens.shape[1] == self.domain_extrema.shape[0])
        extrema_2 = torch.tensor([[-1.0, -0.8],[mu_quake[0]-0.4, mu_quake[0]+0.4], [mu_quake[1]-0.4, mu_quake[1]+0.4]])
        return tens * (extrema_2[:, 1] - extrema_2[:, 0]) + extrema_2[:, 0]

    def add_solid_points1(self):
        input_s = self.convert(self.soboleng.draw(int(self.n_collocation_points/2)))
       # print(input_s.shape)
       # input_s2 = self.convert2(self.soboleng.draw(int(self.n_collocation_points/2)))
       # print(input_s2.shape)

       # input = torch.cat((input_s2, input_s), dim=0)
       # print(input.shape)

        return input_s
    def add_solid_points2(self):
        #input_s = self.convert(self.soboleng.draw(int(self.n_collocation_points)))
       # print(input_s.shape)
        input_s2 = self.convert2(self.soboleng.draw(int(self.n_collocation_points/2)))
       # print(input_s2.shape)

       # input = torch.cat((input_s2, input_s), dim=0)
       # print(input.shape)

        return input_s2

   # def add_init_points(self):
       # input_init = self.convert(self.soboleng.draw(self.n_collocation_points))
       # input_init[:, 0] = torch.full(input_init[:, 0].shape, -1.0)
        # Apply initial conditions
       # x_part = torch.pow(input_init[:, 1] - mu_quake[0], 2)
       # y_part = torch.pow(input_init[:, 2] - mu_quake[1], 2)
       # exponent = -0.5 * torch.pow((torch.sqrt(x_part + y_part+ 1e-8) / sigma_quake), 2)
       # earthquake_spike = torch.exp(exponent)
       # u0x = earthquake_spike
       # u0y = earthquake_spike
       # output_init = torch.zeros([n_points_per_training_set, 2], dtype=torch.float32)
       # output_init[:,0] = u0x
       # output_init[:,1] = u0y
       # return input_init,output_init

    # Function returning the training sets as dataloader
    def assemble_datasets(self):
        input_s1= self.add_solid_points1()
        input_s2 = self.add_solid_points2()
       # input_init, output_init = self.add_init_points()
        training_set_s1 = DataLoader(torch.utils.data.TensorDataset(input_s1),batch_size=int(self.n_collocation_points/2), shuffle=False, num_workers=4)
        training_set_s2 = DataLoader(torch.utils.data.TensorDataset(input_s2),batch_size=int(self.n_collocation_points/2), shuffle=False, num_workers=4)

        #print(type(training_set_s))
        #training_set_init = DataLoader(torch.utils.data.TensorDataset(input_init, output_init),batch_size=self.n_collocation_points, shuffle=False, num_workers=4)
        return training_set_s1,training_set_s2#, training_set_init

    def apply_initial_condition(self, input_init):
        u_pred_init = self.approximate_solution(input_init)
        return u_pred_init

    def compute_solid_loss(self, input_s,epoch):
        U = self.pinn_model_eval(input_s, mu_quake, sigma_quake, solid_boundary=0.0, t0=-1.0)
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

        if epoch%50 == 0:
            residual = residual_solid[0:500]
            print(residual.shape)
            residual = residual.detach().numpy()
            print(residual)
            cm = plt.cm.get_cmap('RdYlBu')
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection='3d')
            p = ax.scatter(input_s[:, 0].detach().numpy(),input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=np.abs(residual), cmap=cm)
            fig.colorbar(p)
            plt.show()

        return loss_solid

    def compute_loss(self,inp_train_s,epoch):
        #loss_init = torch.mean(abs(u_train_init - self.apply_initial_condition(inp_train_init)) ** 2)
        loss_solid = self.compute_solid_loss(inp_train_s,epoch)
        loss = torch.log10(loss_solid)
        wandb.log({"loss": loss.item()})
        wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        #wandb.log({"Init loss": torch.log10(loss_init).item()})
        return loss

   # def compute_loss_init_only(self,inp_train_init,u_train_init):
     #   loss_init = torch.mean(abs(u_train_init - self.apply_initial_condition(inp_train_init)) ** 2)
      #  loss = torch.log10(loss_init)
      ##  wandb.log({"loss": loss.item()})
      #  wandb.log({"Init loss": torch.log10(loss_init).item()})
      #  return loss

    def fit(self, num_epochs, optimizer, verbose=False):

        inp_train_s1= next(iter(self.training_set_s1))[0]
        inp_train_s2 = next(iter(self.training_set_s2))[0]

        print(inp_train_s1)
        print(inp_train_s2)
        inp_train_s1 = inp_train_s1.to(device)
        inp_train_s1.requires_grad = True
        #inp_train_s2 = inp_train_s2.to(device)
       # inp_train_init, u_train_init = next(iter(self.training_set_init))

       # inp_train_init = inp_train_init.to(device)
       # u_train_init = u_train_init.to(device)
        self.approximate_solution = self.approximate_solution.to(device)
        #inp_train_s1.requires_grad = True
        #inp_train_s_combined = torch.cat((inp_train_s1, inp_train_s2), dim=0)
      #  inp_train_init.requires_grad = True
        history = list()
        # Loop over epochs
        for epoch in range(num_epochs):
            #if epoch >= 0.1 * num_epochs:
                #inp_train = inp_train_s_combined.to(device)
                #inp_train.requires_grad = True
            #else:
                #inp_train = inp_train_s1.to(device)
                #inp_train.requires_grad = True


            #self.redraw()
            #inp_train_s = next(iter(self.training_set_s))[0]
            #inp_train_s = inp_train_s.to(device)
            #inp_train_s.requires_grad = True
            if verbose: print("################################ ", epoch, " ################################")
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(inp_train_s1,epoch)
                loss.backward()
                history.append(loss.item())
                return loss

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
                #inp_train_s = next(iter(self.training_set_s))[0]
                #inp_train_s = inp_train_s.to(device)
                #inp_train_s.requires_grad = True
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

wandb.init(project='Semester Thesis',name = 'Print Error test')
pinn = Pinns(n_points_per_training_set)
n_epochs = 200
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                              lr=float(0.8),
                              max_iter=3,
                              max_eval=3,
                              history_size=2000,
                              line_search_fn="strong_wolfe",
                              tolerance_grad=1e-8,
                              tolerance_change=1.0 * np.finfo(float).eps)
hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_LBFGS,
                verbose=True)
torch.save(pinn.approximate_solution.state_dict(),
           "solid_only_my_ansatz_printerror.pth")
