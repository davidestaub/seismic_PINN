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


n_collocation_points = 40000
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
mu_quake = [0, 0]
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
                                              n_hidden_layers=4,
                                              neurons=1024,
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
        #torch.tanh(torch.tanh((t + 1.0) * a)) for a in {2,20}
        #torch.sigmoid(15.0 * (b - ((t + 2.75) * 1.0))) for b in {1.8,2.2}
        U[:, 1] = U_perturbation[:, 1] * torch.tanh(torch.tanh((t + 1.0) * 5)) + u0y * torch.sigmoid(
            15.0 * (2.0 - ((t + 2.75) * 1.0)))
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

    def convert_march(self,tens,time):
        extrema_3 = torch.tensor([[-1.0, time], [-1.0, 1.0], [-1.0, 1.0]])
        return tens * (extrema_3[:, 1] - extrema_3[:, 0]) + extrema_3[:, 0]

    def add_solid_points_march(self,time):
        input_s = self.convert_march(self.soboleng.draw(int(self.n_collocation_points)),time)
        return input_s

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
        input_s1= self.add_solid_points()
        input_st0 = self.add_solid_points_t0()
        #input_s2 = self.add_solid_points2()
       # input_init, output_init = self.add_init_points()
        training_set_s1 = DataLoader(torch.utils.data.TensorDataset(input_s1),batch_size=int(self.n_collocation_points), shuffle=False, num_workers=4)
        training_set_s_t0 = DataLoader(torch.utils.data.TensorDataset(input_st0),batch_size=int(self.n_collocation_points/10), shuffle=False, num_workers=4)

        #print(type(training_set_s))
        #training_set_init = DataLoader(torch.utils.data.TensorDataset(input_init, output_init),batch_size=self.n_collocation_points, shuffle=False, num_workers=4)
        return training_set_s1,training_set_s_t0#, training_set_init

    def apply_initial_condition(self, input_init):
        u_pred_init = self.approximate_solution(input_init)
        return u_pred_init

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

        stress_tensor_00 = lamda_solid * (eps[0, 0] + eps[1, 1]) + 2.0 * mu_solid * eps[0, 0]
        stress_tensor_off_diag = 2.0 * mu_solid * eps[0, 1]
        stress_tensor_11 = lamda_solid * (eps[0, 0] + eps[1, 1]) + 2.0 * mu_solid * eps[1, 1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(input_s.size(0), 2, dtype=torch.float32, device=input_s.device)
        div_stress[:, 0] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[:, 1] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]
        too_close_to_boundary = ((input_s[:, 1] < -0.9) | (input_s[:, 1] > 0.9) | (input_s[:, 2] < -0.9) | (input_s[:, 2] > 0.9))

        # Step 3: Create a mask that's `True` for all points far enough away from the boundary

        # Step 4: Create a new mask for residual_solid by repeating each element of far_from_boundary twice
        mask_resid = torch.cat([too_close_to_boundary, too_close_to_boundary])


        # Compute residuals
        dt2_combined = torch.stack((dt2_x, dt2_y), dim=1)
        residual_solid = rho_solid * dt2_combined - div_stress
        # Step 5: Apply the new mask to your residuals before computing the loss
        residual_solid = residual_solid.reshape(-1, )

        residual_solid[mask_resid] = 0.0
        residual_solid_masked = residual_solid
        # Step 6: Finally, compute your loss only over these residuals
        loss_solid = torch.mean(abs(residual_solid_masked) ** 2)
        #loss_solid = torch.mean(abs(residual_solid) **2)

        return loss_solid

    def get_solid_residual(self,input_s):
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

        stress_tensor_00 = lamda_solid * (eps[0, 0] + eps[1, 1]) + 2.0 * mu_solid * eps[0, 0]
        stress_tensor_off_diag = 2.0 * mu_solid * eps[0, 1]
        stress_tensor_11 = lamda_solid * (eps[0, 0] + eps[1, 1]) + 2.0 * mu_solid * eps[1, 1]
        stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                     torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

        # Compute divergence of the stress tensor
        div_stress = torch.zeros(input_s.size(0), 2, dtype=torch.float32, device=input_s.device)
        div_stress[:, 0] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
        div_stress[:, 1] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                           torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]

        too_close_to_boundary = ((input_s[:, 1] < -0.9) | (input_s[:, 1] > 0.9) | (input_s[:, 2] < -0.9) | (input_s[:, 2] > 0.9))

        # Step 3: Create a mask that's `True` for all points far enough away from the boundary

        # Step 4: Create a new mask for residual_solid by repeating each element of far_from_boundary twice
        mask_resid = torch.cat([too_close_to_boundary, too_close_to_boundary])


        # Compute residuals
        dt2_combined = torch.stack((dt2_x, dt2_y), dim=1)
        residual_solid = rho_solid * dt2_combined - div_stress
        # Step 5: Apply the new mask to your residuals before computing the loss
        residual_solid = residual_solid.reshape(-1, )

        residual_solid[mask_resid] = 0.0
        residual_solid_masked = residual_solid



        U_without_ansatz = self.approximate_solution(input_s)


        return residual_solid_masked,U,U_without_ansatz


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
        #loss_init = torch.mean(abs(u_train_init - self.apply_initial_condition(inp_train_init)) ** 2)
        loss_solid = self.compute_solid_loss(inp_train_s)
        loss_no_init_velocity = self.compute_no_init_velocity_loss(inp_train_s_t0)
        loss = torch.log10(loss_solid + loss_no_init_velocity)
        wandb.log({"loss": loss.item()})
        wandb.log({"Solid loss": torch.log10(loss_solid).item()})
        wandb.log({"no init vel loss": torch.log10(loss_no_init_velocity).item()})
        return loss

   # def compute_loss_init_only(self,inp_train_init,u_train_init):
     #   loss_init = torch.mean(abs(u_train_init - self.apply_initial_condition(inp_train_init)) ** 2)
      #  loss = torch.log10(loss_init)
      ##  wandb.log({"loss": loss.item()})
      #  wandb.log({"Init loss": torch.log10(loss_init).item()})
      #  return loss

    def fit(self, num_epochs, optimizer, verbose=False):

        inp_train_s= next(iter(self.training_set_s))[0]
        #visualize_set = inp_train_s.to(device)
        #visualize_set[:, 0] = -0.99
        #visualize_set.requires_grad = True
        training_set_no_init_vel = next(iter(self.training_set_s_t0))[0]
        training_set_no_init_vel = training_set_no_init_vel.to(device)
        training_set_no_init_vel.requires_grad = True
        self.approximate_solution = self.approximate_solution.to(device)
        training_set_s = inp_train_s.to(device)
        print(training_set_s,training_set_s.shape)
        training_set_s = training_set_s.to(device)
        training_set_s.requires_grad = True


        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            #if epoch%500==0:
                #final_time = float(epoch/num_epochs) - 1.0
                #training_set_s = self.add_solid_points_march(final_time)
                #training_set_s = training_set_s.to(device)
                #training_set_s.requires_grad = True


            #num_samples = min(len(inp_train_s), epoch  + 5000)
            # We create a new DataLoader that only includes the first num_samples of the input_s
            #training_set_s = inp_train_s[:num_samples]
            #training_set_s = training_set_s.to(device)
            #training_set_s.requires_grad = True

            #residual = self.get_solid_residual(visualize_set).detach().cpu().numpy()
            #residual_x = residual[0:int(len(residual)/2)]
            #im = plt.scatter(visualize_set[:,1].detach().cpu().numpy(),visualize_set[:,2].detach().cpu().numpy(),c=residual_x,s=5)
            #plt.show()
            #plt.colorbar(im)
            #wandb.log({"Residual at -0.99": plt})

            if True:
                time_list = [-1.0, -0.99, -0.98,-0.95]
                res_list_ux = []
                res_list_uy = []
                Uwith_list_ux = []
                Uwith_list_uy = []
                Uwithouth_list_ux = []
                Uwithouth_list_uy = []
                numpoints_sqrt = 100


                for i in time_list:
                    inputs = self.soboleng.draw(int(pow(numpoints_sqrt, 2)))
                    grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                                    torch.linspace(-1.0, 1.0, numpoints_sqrt))
                    grid_x = torch.reshape(grid_x, (-1,))
                    grid_y = torch.reshape(grid_y, (-1,))
                    print(i)
                    inputs[:, 1] = grid_x
                    inputs[:, 2] = grid_y
                    inputs[:, 0] = i
                    inputs = inputs.to(device)
                    inputs.requires_grad =True
                    residual,U_with_ansatz,U_withouth_ansatz = self.get_solid_residual(inputs)

                    residual_x = residual[0:int(len(residual) / 2)]
                    residual_y = residual[int(len(residual) / 2):]
                    residual_x_out = residual_x.detach()
                    residual_y_out = residual_y.detach()
                    np_residual_x_out = residual_x_out.cpu().numpy()
                    np_residual_y_out  = residual_y_out .cpu().numpy()
                    B_ux = np.reshape(np_residual_x_out, (-1, int(np.sqrt(np_residual_x_out.shape[0]))))
                    B_uy = np.reshape(np_residual_y_out, (-1, int(np.sqrt(np_residual_y_out.shape[0]))))
                    res_list_ux.append(B_ux)
                    res_list_uy.append(B_uy)

                    Uwithouth_x = U_withouth_ansatz[:, 0]
                    Uwithouth_y = U_withouth_ansatz[:, 1]
                    Uwithouth_x_out = Uwithouth_x.detach()
                    Uwithouth_y_out = Uwithouth_y.detach()
                    np_Uwithouth_x_out = Uwithouth_x_out.cpu().numpy()
                    np_Uwithouth_y_out = Uwithouth_y_out.cpu().numpy()
                    Uwithouth_ux = np.reshape(np_Uwithouth_x_out, (-1, int(np.sqrt(np_Uwithouth_x_out.shape[0]))))
                    Uwithouth_uy = np.reshape(np_Uwithouth_y_out, (-1, int(np.sqrt(np_Uwithouth_y_out.shape[0]))))
                    Uwithouth_list_ux.append(Uwithouth_ux)
                    Uwithouth_list_uy.append(Uwithouth_uy)

                    Uwith_x = U_with_ansatz[:, 0]
                    Uwith_y = U_with_ansatz[:, 1]
                    Uwith_x_out = Uwith_x.detach()
                    Uwith_y_out = Uwith_y.detach()
                    np_Uwith_x_out = Uwith_x_out.cpu().numpy()
                    np_Uwith_y_out = Uwith_y_out.cpu().numpy()
                    Uwith_ux = np.reshape(np_Uwith_x_out, (-1, int(np.sqrt(np_Uwith_x_out.shape[0]))))
                    Uwith_uy = np.reshape(np_Uwith_y_out, (-1, int(np.sqrt(np_Uwith_y_out.shape[0]))))
                    Uwith_list_ux.append(Uwith_ux)
                    Uwith_list_uy.append(Uwith_uy)

                res_ux = np.dstack(res_list_ux)
                res_uy = np.dstack(res_list_uy)
                res_ux = np.rollaxis(res_ux, -1)
                res_uy = np.rollaxis(res_uy, -1)

                Uwith_ux = np.dstack(Uwith_list_ux)
                Uwith_uy = np.dstack(Uwith_list_uy)
                Uwith_ux = np.rollaxis(Uwith_ux, -1)
                Uwith_uy = np.rollaxis(Uwith_uy, -1)

                Uwithouth_ux = np.dstack(Uwithouth_list_ux)
                Uwithouth_uy = np.dstack(Uwithouth_list_uy)
                Uwithouth_ux = np.rollaxis(Uwithouth_ux, -1)
                Uwithouth_uy = np.rollaxis(Uwithouth_uy, -1)

                for i in range(0, len(res_list_ux)):
                    plt.figure(figsize=(10, 6))
                    im_ux = plt.imshow(res_ux[i, :, :])
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.title("residual ux @ time = {}".format(time_list[i]))
                    plt.colorbar(im_ux)
                    wandb.log({"residual ux @ time = {}".format(time_list[i]): plt})
                    plt.figure(figsize=(10, 6))

                    im_uy = plt.imshow(res_uy[i, :, :])
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.title("residual uy @ time = {}".format(time_list[i]))
                    plt.colorbar(im_uy)
                    wandb.log({"residual uy @ time = {}".format(time_list[i]): plt})

                for i in range(0, len(Uwith_list_ux)):
                    plt.figure(figsize=(10, 6))
                    im_ux = plt.imshow(Uwith_ux[i, :, :])
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.title("Uwith ux @ time = {}".format(time_list[i]))
                    plt.colorbar(im_ux)
                    wandb.log({"Uwith ux @ time = {}".format(time_list[i]): plt})
                    plt.figure(figsize=(10, 6))

                    im_uy = plt.imshow(Uwith_uy[i, :, :])
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.title("Uwith uy @ time = {}".format(time_list[i]))
                    plt.colorbar(im_uy)
                    wandb.log({"Uwith uy @ time = {}".format(time_list[i]): plt})

                for i in range(0, len(Uwithouth_list_ux)):
                    plt.figure(figsize=(10, 6))
                    im_ux = plt.imshow(Uwithouth_ux[i, :, :])
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.title("Uwithouth ux @ time = {}".format(time_list[i]))
                    plt.colorbar(im_ux)
                    wandb.log({"Uwithouth ux @ time = {}".format(time_list[i]): plt})
                    plt.figure(figsize=(10, 6))

                    im_uy = plt.imshow(Uwithouth_uy[i, :, :])
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.title("Uwithouth uy @ time = {}".format(time_list[i]))
                    plt.colorbar(im_uy)
                    wandb.log({"Uwithouth uy @ time = {}".format(time_list[i]): plt})


            if verbose: print("################################ ", epoch, " ################################")
            def closure():
                optimizer.zero_grad()
                loss = self.compute_loss(training_set_s,training_set_no_init_vel)
                loss.backward()
                history.append(loss.item())
                return loss

            if epoch % 50 == 0:
                time_list = [-1.0,0.0,-0.95]
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

wandb.init(project='Semester Thesis',name = ' LBFGS center no march big sigma visualize test very wide very expensive')
pinn = Pinns(n_points_per_training_set)
n_epochs = 200
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                              lr=float(1.0),
                              max_iter=400,
                              max_eval=400,
                              history_size=4000,
                              line_search_fn="strong_wolfe",
                              tolerance_grad=1e-8,
                              tolerance_change=1.0 * np.finfo(float).eps)

#optimizer_ADAM = optim.Adam(pinn.approximate_solution.parameters(),
                           # lr=0.01,
                            #)

hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_LBFGS,
                verbose=True)
torch.save(pinn.approximate_solution.state_dict(),
           "test.pth")