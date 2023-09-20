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
import configparser
import initial_conditions
import shutil

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)
config = configparser.ConfigParser()
config.read("config.ini")

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

path_list  = [
"../pre_trained_models/FLIPY_NEW_PINN_explosion_gpu_a01_120000_70_1.0_200_200_800_3_64_tanh_0.7.pth",
#"../pre_trained_models/FLIPY_NEW_PINN_explosion_gpu_a01_80000_30_1.0_200_200_800_3_128_tanh_0.2.pth",
#"../pre_trained_models/FLIPY_NEW_PINN_explosion_gpu_a01_120000_30_1.0_200_200_800_3_128_tanh_0.1.pth",
#"../pre_trained_models/FLIPY_NEW_PINN_explosion_gpu_a01_120000_30_1.0_200_200_800_3_128_tanh_0.07.pth",
#"../pre_trained_models/FLIPY_NEW_PINN_explosion_gpu_a01_120000_30_1.0_200_200_800_3_128_tanh_0.05.pth",
#"../pre_trained_models/FLIPY_NEW_PINN_explosion_gpu_a01_120000_70_1.0_200_200_800_3_64_tanh_0.01.pth"
]


for path in path_list:

    t1 = float(path.split("tanh_")[1].split(".pth")[0])


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
                                                [-1.0, 1.0], [-1.0, 1.0]])  # Space dimension
            self.approximate_solution = NeuralNet(input_dimension=3, output_dimension=2,
                                                  n_hidden_layers=int(config['Network']['n_hidden_layers']),
                                                  neurons=int(config['Network']['n_neurons']),
                                                  regularization_param=0.,
                                                  regularization_exp=2.,
                                                  retrain_seed=3)
            self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
            # self.training_set_s1, self.training_set_s2 = self.assemble_datasets()
            self.training_set_s, self.training_set_s_t0 = self.assemble_datasets()
            # print(type(self.training_set_s))

        def pinn_model_eval(self, input_tensor):
            # Evaluate the model with the given input tensor
            U_perturbation = self.approximate_solution(input_tensor)
            t = input_tensor[:, 0]
            u0x, u0y = initial_conditions.initial_condition_explosion(input_tensor)
            U = torch.zeros_like(U_perturbation)

            #t1 = float(config['initial_condition']['t1'])
            U[:, 0] = U_perturbation[:, 0] * torch.tanh(2.5 *t / t1)**2 + u0x * torch.exp(-0.5 * (1.5 * t/t1)**2)
            U[:, 1] = U_perturbation[:, 1] * torch.tanh(2.5 *t / t1)**2  + u0y * torch.exp(-0.5 * (1.5 * t/t1)**2)
            return U

        def convert(self, tens):
            assert (tens.shape[1] == self.domain_extrema.shape[0])
            return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

        def convert_march(self, tens, time):
            extrema_3 = torch.tensor([[-1.0, time], [-1.0, 1.0], [-1.0, 1.0]])
            return tens * (extrema_3[:, 1] - extrema_3[:, 0]) + extrema_3[:, 0]

        def add_solid_points_march(self, time):
            input_s = self.convert_march(self.soboleng.draw(int(self.n_collocation_points)), time)
            return input_s

        def add_solid_points(self):
            # ADDed sorting
            input_s = self.convert(self.soboleng.draw(int(self.n_collocation_points)))
            sorted_indices = torch.argsort(input_s[:, 0])
            input_s = input_s[sorted_indices]
            return input_s

        def add_solid_points_t0(self):
            input_s = self.convert(self.soboleng.draw(int(self.n_collocation_points / 10)))
            input_s[:, 0] = -1.0
            return input_s

        # Function returning the training sets as dataloader
        def assemble_datasets(self):
            input_s1 = self.add_solid_points()
            input_st0 = self.add_solid_points_t0()
            # input_s2 = self.add_solid_points2()
            # input_init, output_init = self.add_init_points()
            training_set_s1 = DataLoader(torch.utils.data.TensorDataset(input_s1),
                                         batch_size=int(self.n_collocation_points), shuffle=False, num_workers=4)
            training_set_s_t0 = DataLoader(torch.utils.data.TensorDataset(input_st0),
                                           batch_size=int(self.n_collocation_points / 10), shuffle=False, num_workers=4)

            # print(type(training_set_s))
            # training_set_init = DataLoader(torch.utils.data.TensorDataset(input_init, output_init),batch_size=self.n_collocation_points, shuffle=False, num_workers=4)
            return training_set_s1, training_set_s_t0  # , training_set_init

        def apply_initial_condition(self, input_init):
            u_pred_init = self.approximate_solution(input_init)
            return u_pred_init

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

        def compute_no_init_velocity_loss(self, input_st0):
            U = self.pinn_model_eval(input_st0)
            u_s = U[:, :2]  # Solid displacement components (x, y)
            u_x = u_s[:, 0].unsqueeze(1)
            u_y = u_s[:, 1].unsqueeze(1)
            gradient_x = torch.autograd.grad(u_x.sum(), input_st0, create_graph=True)[0]
            gradient_y = torch.autograd.grad(u_y.sum(), input_st0, create_graph=True)[0]
            dt_x = gradient_x[:, 0]
            dt_y = gradient_y[:, 0]

            loss_no_init_velocity_loss = torch.mean(abs(dt_x) ** 2) + torch.mean(abs(dt_y) ** 2)
            return loss_no_init_velocity_loss

        def compute_loss(self, inp_train_s, inp_train_s_t0):
            # loss_init = torch.mean(abs(u_train_init - self.apply_initial_condition(inp_train_init)) ** 2)
            loss_solid = self.compute_solid_loss(inp_train_s)
            loss_no_init_velocity = self.compute_no_init_velocity_loss(inp_train_s_t0)
            loss = torch.log10(loss_solid + loss_no_init_velocity)
            return loss

        def fit(self, num_epochs, optimizer, verbose=False):

            inp_train_s = next(iter(self.training_set_s))[0]
            # visualize_set = inp_train_s.to(device)
            # visualize_set[:, 0] = -0.99
            # visualize_set.requires_grad = True
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

                if config['visualize']['visualize_on']:
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
                else:
                    print(config['visualize'].visualize_on)

                if verbose: print("################################ ", epoch, " ################################")

                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(training_set_s, training_set_no_init_vel)
                    loss.backward()
                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)
            print('Final Loss: ', history[-1])
            return history


    def visualize(model,tag):
        # frames = []
        # fig = plt.figure()
        res_list_ux = []
        res_list_uy = []
        res_list_u = []
        time_list = np.linspace(0,1,100).tolist()
        numpoints_sqrt = 128
        for i in time_list:
            print("i=",i)
            #plt.figure(figsize=(16, 16), dpi=150)
            time = i
            #time_list.append(time)

            inputs = model.soboleng.draw(int(pow(numpoints_sqrt,2)))

            grid_x,grid_y = torch.meshgrid(torch.linspace(-1.0,1.0,numpoints_sqrt),torch.linspace(-1.0,1.0,numpoints_sqrt))
            grid_x = torch.reshape(grid_x,(-1,))
            grid_y = torch.reshape(grid_y, (-1,))


            inputs[:,1] = grid_x
            inputs[:,2] = grid_y
            inputs[:, 0] = time

            ux = model.pinn_model_eval(inputs)[:, 0]
            uy = model.pinn_model_eval(inputs)[:, 1]
            ux_out = ux.detach()
            uy_out = uy.detach()

            np_ux_out = ux_out.numpy()
            np_uy_out = uy_out.numpy()


            B_ux = np.reshape(np_ux_out, (-1, int(np.sqrt(np_ux_out.shape[0]))))
            B_uy = np.reshape(np_uy_out, (-1, int(np.sqrt(np_uy_out.shape[0]))))
            B = np.sqrt(B_uy**2 + B_ux**2)
            res_list_ux.append(B_ux)
            res_list_uy.append(B_uy)
            res_list_u.append(B)


        res_ux =  np.dstack(res_list_ux)
        res_uy = np.dstack(res_list_uy)
        res_ux = np.rollaxis(res_ux, -1)
        res_uy = np.rollaxis(res_uy, -1)
        res_u = np.dstack(res_list_u)
        res_u=np.rollaxis(res_u,-1)
        s_ux = 5 * np.mean(np.abs(res_uy))
        s_uy = 5 * np.mean(np.abs(res_uy))
        s_u = 5 * np.mean(np.abs(res_u))

        test_list_x = []
        test_list_y =[]

        for i in range(0,len(res_list_uy)):
            print(res_ux[i,:,:])

            test_point = res_ux[i,int(128/2)+20,int(128/2)]
            test_list_x.append(time_list[i])
            test_list_y.append(test_point)








            #plt.figure(figsize=(10, 6))
            #im_ux = plt.imshow(res_ux[i,:,:],vmin=-s_ux,vmax=s_ux)
            #plt.xlabel("x")
            #plt.ylabel("y")
            #plt.title("time = {}".format(time_list[i]))
            #plt.colorbar(im_ux)
            #plt.savefig("../images/{}/x/time={}.png".format(tag,i))

            #plt.figure(figsize=(10, 6))
            #im_uy = plt.imshow(res_uy[i, :, :],vmin=-s_uy,vmax=s_uy)
            #plt.xlabel("x")
            #plt.ylabel("y")
            #plt.title("time = {}".format(time_list[i]))
            #plt.colorbar(im_uy)
            #plt.savefig("../images/{}/y/time={}.png".format(tag,i))

            #plt.figure(figsize=(10, 6))
            #im_uy = plt.imshow(res_u[i, :, :],vmin=-s_u,vmax=s_u)
            #plt.xlabel("x")
            #plt.ylabel("y")
            #plt.title("time = {}".format(time_list[i]))
            #plt.colorbar(im_uy)
            #plt.savefig("../images/{}/u/time={}.png".format(tag,i))
            #plt.show()






            #plt.figure(figsize=(10, 6))
            #im_uy = plt.imshow(-res_uy[i, :, :], vmin=-s_uy, vmax=s_uy)
            #plt.xlabel("x uy")
            #plt.ylabel("y uy")
            #plt.title("time = {}".format(time_list[i]))
           # plt.colorbar(im_uy)
            #plt.show()


            print("entryy:", i)

        plt.plot(test_list_x,test_list_y)
        plt.show()


    pinn = Pinns(int(config['Network']['n_points']))
    tag = path.replace(".pth","")

    dir = "../images/"+tag
    # Define the path to the directory
    dir_path = os.path.join(os.getcwd(), dir)

    # Check if the directory already exists
    if os.path.exists(dir_path):
        # Remove the existing directory and all its subdirectories
        shutil.rmtree(dir_path)

    # Create the directory
    os.mkdir(dir_path)

    # Create the subdirectories
    subdirs = ['x', 'y', 'u']
    for subdir in subdirs:
        os.mkdir(os.path.join(dir_path, subdir))

    print(f"Directory '{tag}' created with subdirectories 'x', 'y', and 'u'.")

    n_neurons = int(path.split("_tanh")[0].split("3_")[1])
    my_network = NeuralNet(input_dimension=3, output_dimension=2,n_hidden_layers=3,neurons=n_neurons,regularization_param=0.,regularization_exp=2.,retrain_seed=42)

    my_network.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    pinn.approximate_solution = my_network
    visualize(pinn,tag)
