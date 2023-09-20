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

n_collocation_points = 80000
n_points_per_training_set = int(n_collocation_points/4)

#example values lambda = 20GPa and mu = 30 GPa
#lamda_solid = 20
lamda_solid = 1.0
#mu_solid = 30
mu_solid = 1.0
#taken from internet density of granite = 1463.64kg/m3
#rho_solid = 1463.64
rho_solid = 1.0
#rho_fluid = 997
rho_fluid = 1.0
#c2 = 1450**2
c2 = 1.0
mu_quake = [0, -0.5]
sigma_quake = min(2, 1) * 0.05
#plot residual spatially
#do solid only (with correct parameters)
#resampling based on weight from previous itteration (withouth resampling just reweighting)

# Construct the 4th-order elastic tensor C^U
CU = torch.zeros(2, 2, 2, 2)
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                CU[i, j, k, l] = lamda_solid * (int(i == j) * int(k == l)) + mu_solid * (int(i == k) * int(j == l) + int(i == l) * int(j == k))
print(CU)
CU = CU.to(device)

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
                                              n_hidden_layers=3,
                                              neurons=64,
                                              regularization_param=0.,
                                              regularization_exp=2.,
                                              retrain_seed=42)
        wandb.watch(self.approximate_solution, log_freq=100)
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        self.training_set_s, self.training_set_f, self.training_set_b, self.training_set_init = self.assemble_datasets()

    def pinn_model_eval(self, input_tensor, mu, sigma, solid_boundary=0.0, t0=-1.0):
        # Evaluate the model with the given input tensor
        U_perturbation = self.approximate_solution(input_tensor)
        # Apply initial conditions
        #t = input_tensor[:, 0]
        #x_part = torch.pow(input_tensor[:, 1] - mu[0], 2)
        #y_part = torch.pow(input_tensor[:, 2] - mu[1], 2)
        #exponent = -0.5 * torch.pow((torch.sqrt(x_part + y_part+ 1e-8) / sigma), 2)
        #earthquake_spike = torch.exp(exponent)
        #solid_mask = input_tensor[:, 2] < solid_boundary
        #u0x = earthquake_spike * solid_mask
        #u0y = earthquake_spike * solid_mask
        #p = plt.scatter(input_tensor[:, 1].detach().numpy(),input_tensor[:, 2].detach().numpy(),c=u0x.detach().numpy())
        #plt.colorbar(p)
        #plt.show()
        #(u0x)
        # Apply the initial conditions for each component of the state vector
        #U = torch.zeros_like(U_perturbation)
        #TODO: FIX THIS !
        #MY idea:
        #nn * tanh(2 * tanh(5(2x + 2)))
        ## init * sigmoid(50 * (2- (x+2.95)))
        #U[:, 0] = U_perturbation[:, 0] * torch.tanh(torch.tanh((t + 2.0))) + u0x * torch.sigmoid(5.0 * (2.0 - (t + 2.9)))
        #U[:, 1] = U_perturbation[:, 1] * torch.tanh(torch.tanh((t + 2.0))) + u0y * torch.sigmoid(5.0 * (2.0 - (t + 2.9)))
        #U[:, 0] = u0x + U_perturbation[:, 0] * torch.nn.functional.sigmoid(5 * (t / (-t0) - 1))
        #U[:, 1] = u0y + U_perturbation[:, 1] * torch.nn.functional.sigmoid(5 * (t / (-t0) - 1))
        return U_perturbation

    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    def add_solid_points(self):
        input_s = self.convert(self.soboleng.draw(self.n_collocation_points))
        # Set the desired range for y values in the solid domain
        min_y = -1
        max_y = 0
        # Modify y values for the solid domain
        input_s[:, 2] = input_s[:, 2] * (max_y - min_y) / 2 + (max_y + min_y) / 2
        # Ensure that the maximum y value does not reach 0
        input_s[input_s[:, 2] == 0, 2] -= 1e-5
        output_s = torch.full((input_s.shape[0], 1), 0.0)
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
        input_f[input_f[:, 2] == 0, 2] += 1e-8
        output_f = torch.full((input_f.shape[0], 1), 0.0)
        return input_f,output_f
    def add_boundary_points(self):
        # draw points from range -1 to 1
        input_b = self.convert(self.soboleng.draw(self.n_collocation_points))
        input_b[:, 2] = torch.full(input_b[:, 2].shape, 0.0)
        output_b = torch.full((input_b.shape[0], 1), 0.0)
        return input_b,output_b

    def add_init_points(self):
        input_init = self.convert(self.soboleng.draw(self.n_collocation_points))
        input_init[:, 0] = torch.full(input_init[:, 0].shape, -1.0)
        # Apply initial conditions
        x_part = torch.pow(input_init[:, 1] - mu_quake[0], 2)
        y_part = torch.pow(input_init[:, 2] - mu_quake[1], 2)
        exponent = -0.5 * torch.pow((torch.sqrt(x_part + y_part+ 1e-8) / sigma_quake), 2)
        earthquake_spike = torch.exp(exponent)
        solid_mask = input_init[:, 2] < 0.0
        u0x = earthquake_spike * solid_mask
        u0y = earthquake_spike * solid_mask
        output_init = torch.zeros([n_points_per_training_set, 2], dtype=torch.float32)
        output_init[:,0] = u0x
        output_init[:,1] = u0y
        return input_init,output_init



    # Function returning the training sets as dataloader
    def assemble_datasets(self):
        input_s, output_s = self.add_solid_points()
        input_f, output_f = self.add_fluid_points()
        input_b, output_b = self.add_boundary_points()
        input_init, output_init = self.add_init_points()
        training_set_s = DataLoader(torch.utils.data.TensorDataset(input_s, output_s),batch_size=self.n_collocation_points, shuffle=False, num_workers=4)
        training_set_f = DataLoader(torch.utils.data.TensorDataset(input_f, output_f),batch_size=self.n_collocation_points, shuffle=False, num_workers=4)
        training_set_b = DataLoader(torch.utils.data.TensorDataset(input_b, output_b),batch_size=self.n_collocation_points, shuffle=False, num_workers=4)
        training_set_init = DataLoader(torch.utils.data.TensorDataset(input_init, output_init),batch_size=self.n_collocation_points, shuffle=False, num_workers=4)
        return training_set_s, training_set_f, training_set_b, training_set_init

    def apply_initial_condition(self, input_init):
        u_pred_init = self.approximate_solution(input_init)
        return u_pred_init

    def compute_solid_loss(self, input_s):
        U = self.pinn_model_eval(input_s, mu_quake, sigma_quake, solid_boundary=0.0, t0=-1.0)
        u_s = U[:, :2]  # Solid displacement components (x, y)
        u_x = u_s[:, 0].unsqueeze(1)
        u_y = u_s[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(u_x.sum(), input_s, create_graph=True)[0]
        gradient_y = torch.autograd.grad(u_y.sum(), input_s, create_graph=True)[0]
        #TODO: unneccesary
        gradient = torch.cat((gradient_x, gradient_y), dim=1)
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
        stress_tensor_11 = lamda_solid * (eps[0,0] + eps[1,1]) + 2.0 * mu_solid * eps[1,1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)), torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        #TODO: check indexing
        div_stress = torch.zeros(input_s.size(0), 2, dtype=torch.float32, device=input_s.device)
        div_stress[:, 0] = torch.autograd.grad(stress_tensor[:, 0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[:, 0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[:, 1] = torch.autograd.grad(stress_tensor[:, 1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[:, 1, 1].sum(), input_s, create_graph=True)[0][:, 2]
        # Compute residuals
        dt2_combined = torch.stack((dt2_x, dt2_y), dim=1)
        residual_solid = rho_solid * dt2_combined - div_stress
        residual_solid = residual_solid.reshape(-1, )
        loss_solid = torch.mean(abs(residual_solid) **2)

        return loss_solid

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
        duy_dt2 = torch.autograd.grad(duy_dt.sum(), input_f, create_graph=True)[0][:, 0]
        duy_dx2 = torch.autograd.grad(duy_dx.sum(), input_f, create_graph=True)[0][:, 1]
        duy_dy2 = torch.autograd.grad(duy_dy.sum(), input_f, create_graph=True)[0][:, 2]
        x_residual = dux_dt2 - c2 * (dux_dx2 + dux_dy2)
        y_residual = duy_dt2 - c2 * (duy_dx2 + duy_dy2)
        l_x = torch.mean(abs(x_residual.reshape(-1, )) ** 2)
        l_y = torch.mean(abs(y_residual.reshape(-1, )) ** 2)

        return l_x + l_y

    def compute_boundary_loss(self,input_b):
        U = self.pinn_model_eval(input_b, mu_quake, sigma_quake, solid_boundary=0.0, t0=-1.0)
        # Solid stress:
        u_x = U[:, 0].unsqueeze(1)
        u_y = U[:, 1].unsqueeze(1)
        gradient_x = torch.autograd.grad(u_x.sum(), input_b, create_graph=True)[0]
        gradient_y = torch.autograd.grad(u_y.sum(), input_b, create_graph=True)[0]
        gradient = torch.cat((gradient_x, gradient_y), dim=1)
        dx_x = gradient[:, 1]
        dy_x = gradient[:, 2]
        dx_y = gradient[:, 4]
        dy_y = gradient[:, 5]
        diag_1 = 2.0 * dx_x
        diag_2 = 2.0 * dy_y
        off_diag = dy_x + dx_y
        eps = 0.5 * torch.stack((torch.stack((diag_1, off_diag)), torch.stack((off_diag, diag_2))), dim=1)
        # Double tensor contraction
        stress_tensor = torch.einsum('ijkl,ijm->ikm', CU, eps)
        n = torch.tensor([0, 1], device=input_b.device,dtype=torch.float32)
        t = torch.tensor([1, 0], device=input_b.device,dtype=torch.float32)
        normal_stress_solid = torch.einsum('ijk,j->ik', stress_tensor, n)
        tangent_stress_solid = torch.einsum('ijk,j->ik', stress_tensor, t)
        # FLuid stress:
        # Compute the divergence
        div_u = dx_x + dy_y
        I = torch.eye(2,device=input_b.device)
        # Reshape div_u to match the tensor product operation
        div_u_reshaped = div_u.repeat(2, 2, 1)
        T_u_f = rho_fluid * c2 * div_u_reshaped * I.unsqueeze(2)
        normal_stress_fluid = torch.einsum('ijk,j->ik', T_u_f, n)
        tangent_stress_fluid = torch.einsum('ijk,j->ik', T_u_f, t)
        #BC1:
        r1 = normal_stress_fluid - normal_stress_solid
        l1 = torch.mean(abs(r1) **2)
        #BC2:
        l2 = torch.mean(abs(tangent_stress_fluid) ** 2)
        #BC3:
        l3 = torch.mean(abs(tangent_stress_solid) ** 2)

        return l1 + l2 + l3


    def compute_loss(self,inp_train_s, u_train_s, inp_train_f, u_train_f, inp_train_b,u_train_b,inp_train_init,u_train_init,verbose):
        U = self.pinn_model_eval(inp_train_b, mu_quake, sigma_quake, solid_boundary=0.0, t0=-1.0)
        loss_init = torch.mean(abs(u_train_init - self.apply_initial_condition(inp_train_init)) ** 2)
        loss_solid = self.compute_solid_loss(inp_train_s)
        loss_fluid = self.compute_fluid_loss(inp_train_f)
        loss_boundary = self.compute_boundary_loss(inp_train_b)
        loss = torch.log10(loss_solid + loss_fluid + loss_boundary+ loss_init)
        wandb.log({"loss": loss.item()})
        wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        wandb.log({"Fluid loss": torch.log10(loss_fluid).item()})
        wandb.log({"Boundary loss": torch.log10(loss_boundary).item()})
        wandb.log({"Init loss": torch.log10(loss_init).item()})
        #if verbose: print("Total loss: ", round(loss.item(), 4), "| Solid Loss: ", round(torch.log10(loss_solid).item(), 4),"| Fluid Loss: ", round(torch.log10(loss_fluid).item(), 4),"| Boundary Loss: ", round(torch.log10(loss_boundary).item(), 4))
        return loss

    def compute_loss_init_only(self,inp_train_init,u_train_init,verbose):
        loss_init = torch.mean(abs(u_train_init - self.apply_initial_condition(inp_train_init)) ** 2)
        loss = torch.log10(loss_init)
        wandb.log({"loss": loss.item()})
        wandb.log({"Init loss": torch.log10(loss_init).item()})
        #if verbose: print("Total loss: ", round(loss.item(), 4), "| Solid Loss: ", round(torch.log10(loss_solid).item(), 4),"| Fluid Loss: ", round(torch.log10(loss_fluid).item(), 4),"| Boundary Loss: ", round(torch.log10(loss_boundary).item(), 4))
        return loss

    def fit(self, num_epochs, optimizer, verbose=False):
        inp_train_s, u_train_s = next(iter(self.training_set_s))
        inp_train_f, u_train_f = next(iter(self.training_set_f))
        inp_train_b, u_train_b = next(iter(self.training_set_b))
        inp_train_init, u_train_init = next(iter(self.training_set_init))
        inp_train_s = inp_train_s.to(device)
        u_train_s = u_train_s.to(device)
        inp_train_f = inp_train_f.to(device)
        u_train_f = u_train_f.to(device)
        inp_train_b = inp_train_b.to(device)
        inp_train_init = inp_train_init.to(device)
        u_train_init = u_train_init.to(device)
        self.approximate_solution = self.approximate_solution.to(device)
        inp_train_s.requires_grad = True
        inp_train_f.requires_grad = True
        inp_train_b.requires_grad = True
        inp_train_init.requires_grad = True
        history = list()
        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")
            if epoch <= num_epochs*0.1:
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss_init_only(inp_train_init, u_train_init, verbose=verbose)
                    loss.backward()
                    history.append(loss.item())
                    return loss
            else:
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_s, u_train_s, inp_train_f, u_train_f, inp_train_b, u_train_b,inp_train_init, u_train_init, verbose=verbose)
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
            optimizer.step(closure=closure)
        print('Final Loss: ', history[-1])
        return history

wandb.init(project='Semester Thesis',name = ' NO ansatz test, epochs 2000, rolled out stress tensor')
pinn = Pinns(n_points_per_training_set)
n_epochs = 2000
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                              lr=float(0.1),
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
           "no_ansatz_rolled_out.pth")

