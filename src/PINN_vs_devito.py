
import torch
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)
import torch.nn as nn
import os
import configparser
import initial_conditions
import shutil
import numpy as np
from cached_property import cached_property
from devito import *
import matplotlib.pyplot as plt
from matplotlib import cm
from sympy import pprint
import torch
from devito.builtins import initialize_function
import PINNs
import FD_devito
import sys
import time


#TODO: create a test routine to run an experiment to test all above models for increasing difficulty
#TODO create comparision pipeline
#TODO: clean up and test devito code
#TODO: turn devito code into an importable module and make it easy to run different test
#TODO: implement test loss for training

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(42)
config = configparser.ConfigParser()


#Changable parameters:

#Number of points along x and y axis, total number of points will be numpoints_sqrt**2
numpoints_sqrt = 256
dt=1.1e-3
mu_quake_x = 0.476
mu_quake_y = 0.1628


mu_quake = [mu_quake_x, mu_quake_y]
mu_quake = torch.tensor(mu_quake)

#pth = "../pre_trained_models/FLIPY_NEW_PINN_conditioned_300000_100_1.0_200_200_800_6_64_tanh_0.07.pth"


if len(sys.argv) < 2:
    raise Exception("Please provide an input folder containing a model path and a config file")
input_folder = sys.argv[1]

config.read("../pre_trained_models/"+input_folder+"/config.ini")
t1 = float(config['initial_condition']['t1'])

# Parameters
rho_solid = float(config['parameters']['rho_solid'])


model_path = "../pre_trained_models/"+input_folder+"/model.pth"
str_path = model_path.replace("../pre_trained_models/","")
model_type = config["parameters"]["model_type"]



pinn_domain_extent_x = [float(config["domain"]["xmin"]),float(config["domain"]["xmax"])]
pinn_domain_extent_y = [float(config["domain"]["ymin"]),float(config["domain"]["ymax"])]
pinn_length_x = pinn_domain_extent_x[1] - pinn_domain_extent_x[0]
pinn_length_y = pinn_domain_extent_y[1] - pinn_domain_extent_y[0]

enlarge_factor = 3
devito_length_x = enlarge_factor*pinn_length_x
devito_length_y = enlarge_factor*pinn_length_y
extent = (devito_length_x,devito_length_y)
devito_center = [devito_length_x/2.0,devito_length_y/2.0]

mu_quake_devito=[devito_center[0]+ mu_quake[0].numpy(),devito_center[1]+ mu_quake[1].numpy()]


#increase Devito domain to avoid Boundary issues (reflections)
shape = (numpoints_sqrt*enlarge_factor, numpoints_sqrt*enlarge_factor)
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))
y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=extent[1]/(shape[1]-1)))
grid = Grid(extent=extent, shape=shape, dimensions=(x,y))
spacing = (extent[0]/shape[0],extent[0]/shape[0])
X, Y = np.meshgrid(np.linspace(0, numpoints_sqrt * spacing[0], numpoints_sqrt), np.linspace(0, numpoints_sqrt * spacing[1], numpoints_sqrt))
print(X.shape,type(X),config["parameters"]["lambda_solid"])

if model_type=="mixture":
    # Generate mixtures for mu and lambda
    mu_mixture = FD_devito.generate_mixture().numpy()
    lambda_mixture = FD_devito.generate_mixture().numpy()
    lambda_solid = FD_devito.compute_param_np(X, Y, lambda_mixture)
    mu_solid = FD_devito.compute_param_np(X, Y, mu_mixture)
elif model_type=="constant":
    lambda_solid = np.full(X.shape, config["parameters"]["lambda_solid"])
    mu_solid = np.full(X.shape, config["parameters"]["mu_solid"])
else:
    raise Exception("model type {} not implemented".format(model_type))



# Create a larger array (7x7)
X_large, Y_large = numpoints_sqrt * enlarge_factor, numpoints_sqrt * enlarge_factor
lambda_large = np.full((X_large, Y_large), 0.0)
# Calculate the position to place the smaller array at the center of the larger array
x_offset = (X_large - numpoints_sqrt) // 2
y_offset = (Y_large - numpoints_sqrt) // 2

# Place the smaller array in the center of the larger array
lambda_large[x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt] = lambda_solid
lambda_large[0:x_offset, :] = lambda_large[x_offset, :]
lambda_large[x_offset + numpoints_sqrt:, :] = lambda_large[x_offset + (numpoints_sqrt - 1), :]
lambda_large[:, 0:y_offset] = lambda_large[:, y_offset][:, np.newaxis]
lambda_large[:, y_offset + numpoints_sqrt:] = lambda_large[:, y_offset + (numpoints_sqrt - 1)][:, np.newaxis]

lambda_solid = lambda_large

Xp, Yp = np.meshgrid(np.linspace(0, (numpoints_sqrt * enlarge_factor) * spacing[0], numpoints_sqrt * enlarge_factor), np.linspace(0, (numpoints_sqrt * enlarge_factor) * spacing[1], numpoints_sqrt * enlarge_factor))
plt.scatter(Xp,Yp,c=lambda_solid)
plt.show()
mu_large = np.full((X_large, Y_large), 0.0)
mu_large[x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt] = mu_solid
mu_large[0:x_offset, :] = mu_large[x_offset, :]
mu_large[x_offset + numpoints_sqrt:, :] = mu_large[x_offset + (numpoints_sqrt - 1), :]
mu_large[:, 0:y_offset] = mu_large[:, y_offset][:, np.newaxis]
mu_large[:, y_offset + numpoints_sqrt:] = mu_large[:, y_offset + (numpoints_sqrt - 1)][:, np.newaxis]
mu_solid = mu_large

#Using both ux and uy as separate field for easier comparison to PINNs and easier pde formulation
ux_devito = TimeFunction(name='ux_devito', grid=grid, space_order=4, time_order=2)
uy_devito = TimeFunction(name='uy_devito', grid=grid, space_order=4, time_order=2)

lambda_ = Function(name='lambda_f', grid=grid, space_order=4)
mu_ = Function(name='mu_f',grid=grid,space_order=4)
initialize_function(lambda_, lambda_solid,nbl=0)
initialize_function(mu_, mu_solid,nbl=0)
print(mu_.data,mu_.data.shape)
plt.scatter(Xp,Yp,c=lambda_.data)
plt.show()
plt.scatter(Xp,Yp,c=mu_.data)
plt.show()


# The initial condition that also was used for The PINN, mimics a perfectly isotropic explosion
u0x_devito, u0y_devito = FD_devito.initial_condition_double_gaussian_derivative(sigma=float(config["parameters"]["sigma_quake"]), mu_quake=mu_quake_devito, nx=shape[0], ny=shape[1], dx=spacing[0], dy=spacing[1])

# Initialize the VectorTimeFunction with the initial values
# Setting both u(t0,x,y) and u(t0+dt,x,y) to u0 to guarantee du0/dt=0
# and because 2nd order time discretization is neeed
ux_devito.data[0] = u0x_devito
uy_devito.data[0] = u0y_devito
ux_devito.data[1] = u0x_devito
uy_devito.data[1] = u0y_devito

FD_devito.plot_field(ux_devito.data[0])
FD_devito.plot_field(uy_devito.data[0])
#Plot from initial field look good
print(ux_devito.shape,uy_devito.shape)

print("start stress calculation")
# Divergence of stress tensor for ux
div_stress_ux = (lambda_ + 2.0 * mu_) * ux_devito.dx2 + mu_ * ux_devito.dy2 + lambda_ * uy_devito.dy.dx + mu_ * uy_devito.dx.dy

# Divergence of stress tensor for uy
div_stress_uy = (lambda_ + 2.0 * mu_) * uy_devito.dy2 + mu_ * uy_devito.dx2 + lambda_ * ux_devito.dx.dy + mu_ * ux_devito.dy.dx

print("stress calculated")
# Elastic wave equation for ux and uy
#print("model damp = ",model.damp,model.damp.data)
pde_x = rho_solid * ux_devito.dt2 - div_stress_ux
pde_y = rho_solid * uy_devito.dt2 - div_stress_uy


#BOundary conditions:
x, y = grid.dimensions
t = grid.stepping_dim
ny,nx = shape[0],shape[1]
bc = [Eq(ux_devito[t+1,x, 0], 0.)]
bc += [Eq(ux_devito[t+1,x, ny-1], 0.)]
bc += [Eq(ux_devito[t+1,0, y], 0.)]
bc += [Eq(ux_devito[t+1,nx-1, y], 0.)]

bc += [Eq(ux_devito[t+1,x, 1], 0.)]
bc += [Eq(ux_devito[t+1,x, ny-2], 0.)]
bc += [Eq(ux_devito[t+1,1, y], 0.)]
bc += [Eq(ux_devito[t+1,nx-2, y], 0.)]

bc += [Eq(uy_devito[t+1,x, 0], 0.)]
bc += [Eq(uy_devito[t+1,x, ny-1], 0.)]
bc += [Eq(uy_devito[t+1,0, y], 0.)]
bc += [Eq(uy_devito[t+1,nx-1, y], 0.)]

bc += [Eq(uy_devito[t+1,x, 1], 0.)]
bc += [Eq(uy_devito[t+1,x, ny-2], 0.)]
bc += [Eq(uy_devito[t+1,1, y], 0.)]
bc += [Eq(uy_devito[t+1,nx-2, y], 0.)]


#Formulating stencil to solve for u forward
stencil_x = Eq(ux_devito.forward,solve(pde_x,ux_devito.forward))
stencil_y = Eq(uy_devito.forward,solve(pde_y,uy_devito.forward))
pprint(stencil_x)
pprint(stencil_y)
op = Operator([stencil_x]+[stencil_y]+bc)

time= 0
index = 0
print("start")


print("Model type not specified, reading from config file instead")
model_type = config['Network']['model_type']
if model_type == "Pinns":
    pinn = PINNs.Pinns(int(config['Network']['n_points']), wandb_on=False, config=config)
elif model_type == "Global_NSources_Conditioned_Pinns":
    pinn = PINNs.Global_NSources_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False, config=config)
elif model_type == "Relative_Distance_NSources_Conditioned_Pinns":
    pinn = PINNs.Relative_Distance_NSources_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                              config=config)
elif model_type == "Relative_Distance_FullDomain_Conditioned_Pinns":
    pinn = PINNs.Relative_Distance_FullDomain_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                                config=config)
elif model_type == "Global_NSources_Conditioned_Lame_Pinns":
    pinn = PINNs.Global_NSources_Conditioned_Lame_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                        config=config)
elif model_type == "Global_FullDomain_Conditioned_Pinns":
    pinn = PINNs.Global_FullDomain_Conditioned_Pinns(int(config['Network']['n_points']), wandb_on=False,
                                                        config=config)
else:
    raise Exception("Model type {} not supported".format(model_type))

tag = str_path.replace(".pth", "")
tag = tag.replace("/model","")

dir = "../images/" + tag
# Define the path to the directory
dir_path = os.path.join(os.getcwd(), dir)
print(dir_path)

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

n_neurons = int(config['Network']['n_neurons'])
n_hidden_layers = int(config['Network']['n_hidden_layers'])
print(n_neurons,n_hidden_layers)
if config['Network']['activation'] == 'tanh':
    activation = nn.Tanh()
else:
    print("unknown activation function", config['Network'].activation)
    exit()
my_network = PINNs.NeuralNet(input_dimension=5, output_dimension=2, n_hidden_layers=n_hidden_layers, neurons=n_neurons,
                       regularization_param=0., regularization_exp=2., retrain_seed=42,activation=nn.Tanh())

my_network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
pinn.approximate_solution = my_network

res_list_ux = []
res_list_uy = []
res_list_u = []
res_list_devito_x = []
res_list_devito_y = []
res_list_devito_u = []
time_list = np.linspace(0, 1, 101).tolist()
devito_time = 0.0
for i in time_list:
    res_list_devito_x.append(np.transpose(ux_devito.data[1][x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt]).copy())
    res_list_devito_y.append(np.transpose(uy_devito.data[1][x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt]).copy())
    res_list_devito_u.append(np.sqrt(np.transpose(uy_devito.data[1][x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt]).copy()**2 + np.transpose(ux_devito.data[1][x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt]).copy()**2))

    op(time_M=10, dt=dt)


    print("i=", i)
    time = i
    soboleng = torch.quasirandom.SobolEngine(dimension=pinn.domain_extrema.shape[0])
    inputs = soboleng.draw(int(pow(numpoints_sqrt, 2)))

    grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                    torch.linspace(-1.0, 1.0, numpoints_sqrt))
    grid_x = torch.reshape(grid_x, (-1,))
    grid_y = torch.reshape(grid_y, (-1,))


    inputs[:, 1] = grid_x
    inputs[:, 2] = grid_y
    inputs[:, 0] = time
    inputs[:, 3] = mu_quake[0]
    inputs[:, 4] = mu_quake[1]

    if model_type == "Relative_Distance_NSources_Conditioned_Pinns" or model_type == "Relative_Distance_FullDomain_Conditioned_Pinns":
        inputs[:, 1] = inputs[:, 1] - inputs[:, 3]
        inputs[:, 2] = inputs[:, 2] - inputs[:, 4]


    ux = pinn.pinn_model_eval(inputs)[:, 0]
    uy = pinn.pinn_model_eval(inputs)[:, 1]
    ux_out = ux.detach()
    uy_out = uy.detach()

    np_ux_out = ux_out.numpy()
    np_uy_out = uy_out.numpy()

    B_ux = np.reshape(np_ux_out, (-1, int(np.sqrt(np_ux_out.shape[0]))))
    B_uy = np.reshape(np_uy_out, (-1, int(np.sqrt(np_uy_out.shape[0]))))
    B = np.sqrt(B_uy ** 2 + B_ux ** 2)
    res_list_ux.append(B_ux)
    res_list_uy.append(B_uy)
    res_list_u.append(B)

res_ux = np.dstack(res_list_ux)
res_uy = np.dstack(res_list_uy)
res_ux = np.rollaxis(res_ux, -1)
res_uy = np.rollaxis(res_uy, -1)
res_u = np.dstack(res_list_u)
res_u = np.rollaxis(res_u, -1)
s_ux = 5 * np.mean(np.abs(res_uy))
s_uy = 5 * np.mean(np.abs(res_uy))
s_u = 5 * np.mean(np.abs(res_u))

#res_devito = np.dstack(res_list_devito)
#res_devito = np.rollaxis(res_devito, -1)

f, axarr = plt.subplots(3, 3, figsize=(15, 20))
plt.subplots_adjust(hspace=-0.1, wspace=0.1)

RMSE = 0
RRMSE = 0
index = 1
for h in range(0, len(res_list_uy)):
    diffx = ((res_uy[h, :, :]) - (res_list_devito_x[h]))
    diffx[0:9, :] = 0
    diffx[:, 0:9] = 0
    diffx[:, numpoints_sqrt-9:numpoints_sqrt] = 0
    diffx[numpoints_sqrt-9:numpoints_sqrt:, :] = 0
    im1x = axarr[0][0].imshow(res_uy[h, :, :], 'bwr', vmin=-2*s_uy, vmax=2*s_uy)
    im2x = axarr[0][1].imshow(res_list_devito_x[h], 'bwr', vmin=-2*s_uy, vmax=2*s_uy)
    im3x = axarr[0][2].imshow(diffx, 'bwr', vmin=-2*s_uy, vmax=2*s_uy)
    RMSE = RMSE + np.sqrt(np.mean(diffx ** 2))
    RRMSE = RRMSE + np.sqrt(np.mean(diffx ** 2))/np.sqrt(np.mean(res_list_devito_x[h] ** 2))
    print("RMSE:", RMSE/index, "RRMSE:", RRMSE/index)
    index = index +1

    diffy = (res_ux[h, :, :]) - (res_list_devito_y[h])
    diffy[0:9, :] = 0
    diffy[:, 0:9] = 0
    diffy[:, numpoints_sqrt-9:numpoints_sqrt]  = 0
    diffy[numpoints_sqrt-9:numpoints_sqrt:, :] = 0
    im1y = axarr[1][0].imshow((res_ux[h, :, :]), 'bwr', vmin=-2 * s_uy, vmax=2 * s_uy)
    im2y= axarr[1][1].imshow((res_list_devito_y[h]), 'bwr', vmin=-2 * s_uy, vmax=2 * s_uy)
    im3y = axarr[1][2].imshow(diffy, 'bwr', vmin=-2 * s_uy, vmax=2 * s_uy)

    diffu = ((res_u[h, :, :]) - (res_list_devito_u[h]))
    diffu[0:9, :] = 0
    diffu[:, 0:9] = 0
    diffu[:, numpoints_sqrt-9:numpoints_sqrt]  = 0
    diffu[numpoints_sqrt-9:numpoints_sqrt:, :] = 0
    im1u = axarr[2][0].imshow(res_u[h, :, :], 'bwr', vmin=-2 * s_uy, vmax=2 * s_uy)
    im2u = axarr[2][1].imshow(res_list_devito_u[h], 'bwr', vmin=-2 * s_uy, vmax=2 * s_uy)
    im3u = axarr[2][2].imshow(diffu, 'bwr', vmin=-2 * s_uy, vmax=2 * s_uy)



    axarr[0][0].set_title("PINN", fontsize=25, pad=20)
    axarr[0][1].set_title("Devito", fontsize=25, pad=20)
    axarr[0][2].set_title("Difference", fontsize=25, pad=20)
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im3x, cax=cbar_ax)
    f.show()
    f.savefig("../images/{}/x/time={}.png".format(tag, h))
    print("entryy:", h)
RMSE = RMSE/len(res_list_uy)
RRMSE = RRMSE/len(res_list_uy)
print("RMSE:",RMSE,"RRMSE:",RRMSE)
