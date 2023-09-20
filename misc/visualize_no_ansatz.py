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
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])

    def pinn_model_eval(self, input_tensor, mu, sigma, solid_boundary=0.0, t0=-1.0):
        # Evaluate the model with the given input tensor
        U_perturbation = self.approximate_solution(input_tensor)

        return U_perturbation


def visualize(model):
    # frames = []
    # fig = plt.figure()
    res_list_ux = []
    res_list_uy = []
    time_list = np.linspace(-1,0,10).tolist()
    numpoints_sqrt = 512
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

        torch.set_printoptions(threshold=10_000)
        print(inputs)
        ux = model.pinn_model_eval(inputs, mu_quake, sigma_quake, solid_boundary=0.0, t0=-1.0)[:, 0]
        uy = model.pinn_model_eval(inputs, mu_quake, sigma_quake, solid_boundary=0.0, t0=-1.0)[:, 1]
        ux_out = ux.detach()
        uy_out = uy.detach()

        np_ux_out = ux_out.numpy()
        np_uy_out = uy_out.numpy()



        B_ux = np.reshape(np_ux_out, (-1, int(np.sqrt(np_ux_out.shape[0]))))
        B_uy = np.reshape(np_uy_out, (-1, int(np.sqrt(np_uy_out.shape[0]))))
        res_list_ux.append(B_ux)
        res_list_uy.append(B_uy)

    res_ux =  np.dstack(res_list_ux)
    res_uy = np.dstack(res_list_uy)
    res_ux = np.rollaxis(res_ux, -1)
    res_uy = np.rollaxis(res_uy, -1)
    s_ux = 5 * np.mean(np.abs(res_ux))
    s_uy = 5 * np.mean(np.abs(res_uy))

    for i in range(0,len(res_list_ux)):
        plt.figure(figsize=(10, 6))
        im_ux = plt.imshow(res_uy[i,:,:])
        print(res_uy[i,:,:])
        plt.xlabel("x uy")
        plt.ylabel("y uy")
        plt.title("time = {}".format(time_list[i]))
        plt.colorbar(im_ux)
        plt.show()





        #plt.figure(figsize=(10, 6))
        #im_uy = plt.imshow(-res_uy[i, :, :], vmin=-s_uy, vmax=s_uy)
        #plt.xlabel("x uy")
        #plt.ylabel("y uy")
        #plt.title("time = {}".format(time_list[i]))
       # plt.colorbar(im_uy)
        #plt.show()


        print("entryy:", i)

n_int = 500
n_tb = 500
n_tb_upsample = 20000


pinn = Pinns(n_int)


my_network = NeuralNet(input_dimension=3, output_dimension=2,n_hidden_layers=3,neurons=64,regularization_param=0.,regularization_exp=2.,retrain_seed=42)

all_ones_path = '../pre_trained_models/no_ansatz_test_1.pth'
my_network.load_state_dict(torch.load(all_ones_path,map_location=torch.device('cpu')))
pinn.approximate_solution = my_network
print(pinn.approximate_solution)
visualize(pinn)
