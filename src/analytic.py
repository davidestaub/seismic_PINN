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
from torch.distributions import MultivariateNormal
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


n_collocation_points = 100000
n_points_per_training_set = int(n_collocation_points)

domain_extrema = torch.tensor([[-1.0, -0.5],[-0.3, 0.3], [-0.3, 0.3]])
soboleng = torch.quasirandom.SobolEngine(dimension=domain_extrema.shape[0])


def convert(tens):
    assert (tens.shape[1] == domain_extrema.shape[0])
    return tens * (domain_extrema[:, 1] - domain_extrema[:, 0]) + domain_extrema[:, 0]

input_s = convert(soboleng.draw(int(n_collocation_points)))

lamda_solid = torch.tensor(2.0)#2.0 * 1e+8
mu_solid = torch.tensor(3.0)#3.0 * 1e+8
rho_solid = torch.tensor(1.0)#1000.0
rho_fluid = torch.tensor(1.0)
mu_quake = torch.tensor([0, 0])
torch.pi = torch.acos(torch.zeros(1)).item() * 2

def normalized_gaussian(x, mean, cov, time, decay_rate=0.01):
    dim = mean.shape[0]
    cov_det = torch.det(cov)
    cov_inv = torch.inverse(cov)

    # The peak height now depends on the time.
    norm_const_time_independent = 1. / (torch.sqrt((2 * torch.pi) ** dim * cov_det))
    norm_const = norm_const_time_independent * torch.exp(-decay_rate * time)
    norm_const = torch.squeeze(norm_const)

    x_sub_mean = x - mean

    # We compute the exponent without any time factor.
    exponent = -0.5 * (x_sub_mean @ cov_inv * x_sub_mean).sum(dim=1)

    return norm_const * torch.exp(exponent)

def smoothstep(edge0, edge1, x):
    t = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def radial_basis_function2(x, mean, radius, time, decay_rate,rbf_sigma):
    # Calculate squared Euclidean distance of each point in x from the center.
    radius = 0.4
    decay_rate=10
    time = time+1
    squared_distance = ((x - mean) ** 2).sum(dim=1)

    # Calculate height based on time.
    height = torch.exp(-decay_rate * time).squeeze()

    # Use smoothstep to create a mask: height inside the ball, smoothly decaying to 0 outside.
    mask = height * smoothstep(radius**2, (radius + decay_rate)**2, squared_distance)

    # Result is mask.
    return mask

def radial_basis_function(x, mean, radius, time, decay_rate, rbf_sigma=0.1):
    # Calculate squared Euclidean distance of each point in x from the center.
    time = time+1
    squared_distance = ((x - mean) ** 2).sum(dim=1)
    print(squared_distance)
    radius = 0.4
    decay_rate = 10

    # RBF value at each point in x.
    rbf = torch.exp(-squared_distance*7)

    # Ball mask: 1.0 inside the ball, smoothly decaying to 0 outside.
    mask = torch.where(squared_distance <= radius ** 2, 1.0, rbf)
    print(mask)

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

def modified_radial_basis_function3(x, mean, radius, time, decay_rate, rbf_sigma=0.1):
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
    # Use sigmoid function for smooth transition to 0.
    # Note: You will need to adjust the parameters inside the sigmoid function to get desired behavior.
    height = 1 / (1 + torch.exp(decay_rate * (time - 0.1)))

    # Result is height times mask.
    result = mask * height

    return result

mean2 = torch.zeros(2)
mean2[0] = mu_quake[0]
mean2[1] = mu_quake[1]



# Create a 3D MultivariateNormal distribution
#dist = MultivariateNormal(mean.to(device), covariance_matrix.to(device))

theta = torch.tensor(0.25 * torch.pi)
# p wave speed
alpha = torch.sqrt((lamda_solid + 2.0 * mu_solid) / rho_solid)
# s wave speed
beta = torch.sqrt(mu_solid / rho_solid)

T=torch.tensor(0.01)
M0 = torch.tensor(0.1)


def pinn_model_eval(input_tensor):
    offset=0.0

    t = input_tensor[:, 0]
    x = input_tensor[: ,1]
    y = input_tensor[: ,2]

    r_abs = torch.sqrt(torch.pow((x - mu_quake[0]) ,2) + torch.pow((y - mu_quake[1]) ,2))
    r_abs = r_abs + 1e-20
    r_hat_x = (x - mu_quake[0] ) /r_abs
    r_hat_y = (y - mu_quake[1] ) /r_abs
    phi_hat_x = -1.0 * r_hat_y
    phi_hat_y = r_hat_x
    #phi = torch.acos( (x - mu_quake[0]) /(torch.sqrt( (x - mu_quake[0])**2 + (y - mu_quake[1])** 2 + 1e-5)))
    phi = torch.atan2(y-mu_quake[1],x - mu_quake[0])
    #mask = phi < 0
    #phi[mask] += 2.0 * np.pi


    M0_dot_input1 = (t + 1 + offset) - r_abs / alpha
    M0_dot_input2 = (t + 1 + offset) - r_abs / beta


    #M0_dot1 = -(torch.exp(scale * -torch.pow(M0_dot_input1, 2) / 2.0) * (M0_dot_input1)) / np.sqrt(2.0 * np.pi)
    #M0_dot2 = -(torch.exp(scale * -torch.pow(M0_dot_input2, 2) / 2.0) * (M0_dot_input2)) / np.sqrt(2.0 * np.pi)
    M_dot1 = M0/(T**2) * (M0_dot_input1 - 3.0*T/2.0) * torch.exp(-(M0_dot_input1- 3.0*T/2.0)**2/T**2)
    M_dot2 = M0/(T**2) * (M0_dot_input2- 3.0*T/2.0) * torch.exp(-(M0_dot_input2- 3.0*T/2.0)**2/T**2)

    M_1 = -(M0/T) * (M0_dot_input1 + T) * torch.exp(-M0_dot_input1/T)
    M_2 = -(M0/T) * (M0_dot_input2 + T) * torch.exp(-M0_dot_input2/T)

    #M0_1 = torch.clamp(M0_1,min=0.0)
    #M0_2 = torch.clamp(M0_2, min=0.0)

    #M_dot1 = torch.clamp(M_dot1, min=0.0)
    #M_dot2 = torch.clamp(M_dot2, min=0.0)

    A_FP_x = torch.sin(2.0 * theta) * torch.cos(phi) * r_hat_x
    A_FP_y = torch.sin(2.0 * theta) * torch.cos(phi) * r_hat_y

    A_FS_x = -torch.cos(theta) * torch.sin(phi) * phi_hat_x
    A_FS_y = -torch.cos(theta) * torch.sin(phi) * phi_hat_y

    A_IP_x = 4.0 *torch.sin(2.0 * theta) * torch.cos(phi) * r_hat_x - 2.0 * (0.0 - torch.cos(theta)*torch.sin(phi) * phi_hat_x)
    A_IP_y = 4.0 * torch.sin(2.0 * theta) * torch.cos(phi) * r_hat_y - 2.0 * (0.0 - torch.cos(theta)*torch.sin(phi) * phi_hat_y)

    A_IS_x = -3.0 * torch.sin(2.0 * theta) *torch.cos(phi) * r_hat_x + 3.0 * (0.0 - torch.cos(theta)*torch.sin(phi) * phi_hat_x)
    A_IS_y = -3.0 * torch.sin(2.0 * theta) *torch.cos(phi) * r_hat_y + 3.0 * (0.0 - torch.cos(theta)*torch.sin(phi) * phi_hat_y)

    intermediate_field_x = (1.0 / (4.0 * torch.pi * alpha ** 2)) * A_IP_x * (1.0/r_abs**2) * M_1 + (1.0 / (4.0 * torch.pi * beta ** 2)) * A_IS_x * (1.0/r_abs**2) * M_2
    intermediate_field_y = (1.0 / (4.0 * torch.pi * alpha ** 2)) * A_IP_y * (1.0/r_abs**2) * M_1 + (1.0 / (4.0 * torch.pi * beta ** 2)) * A_IS_y * (1.0/r_abs**2) * M_2

    intermediate_field_x_simple = A_IP_x * (1.0/r_abs**2) * M_1 + A_IS_x * (1.0/r_abs**2) * M_2
    intermediate_field_y_simple = A_IP_y * (1.0/r_abs**2) * M_1 + A_IS_y * (1.0/r_abs**2) * M_2

    far_field_x = (1.0 / (4.0 * torch.pi * alpha ** 3)) * A_FP_x * (1.0 / r_abs) * M_dot1 + (
                1.0 / (4.0 * torch.pi * beta ** 3)) * A_FS_x * (1.0 / r_abs) * M_dot2

    far_field_x_simple =  A_FP_x * (1.0 / r_abs) * M_dot1 +  A_FS_x * (1.0 / r_abs) * M_dot2

    far_field_y = (1.0 / (4.0 * torch.pi * alpha ** 3)) * A_FP_y * (1.0 / r_abs) * M_dot1 + (
                1.0 / (4.0 * torch.pi * beta ** 3)) * A_FS_y * (1.0 / r_abs) * M_dot2

    far_field_y_simple = A_FP_y * (1.0 / r_abs) * M_dot1 + A_FS_y * (1.0 / r_abs) * M_dot2


    analytic_x = far_field_x #+ intermediate_field_x
    analytic_y = far_field_y #+ intermediate_field_y
    print(analytic_x)

    #analytic_x = A_FP_x * (1.0 / r_abs) * M0_dot1 + A_FS_x * (1.0 / r_abs) * M0_dot2
    #analytic_y = A_FP_y * (1.0 / r_abs) * M0_dot1 + A_FS_y * (1.0 / r_abs) * M0_dot2
    return [[analytic_x,analytic_y,M_dot1,M_dot2,A_FP_x,A_FP_y,A_FS_x,A_FS_y,M0_dot_input1,M0_dot_input2,phi,r_hat_x,r_hat_y,phi_hat_x,phi_hat_y,M_1,M_2],["analytic_x","analytic_y","M0_dot1","M0_dot2","A_FP_x","A_FP_y","A_FS_x","A_FS_y","M0_dot_input1","M0_dot_input2","phi","r_hat_x","r_hat_y","phi_hat_x","phi_hat_y","M0_1","M0_2"]]

index = 0
for j in np.linspace(-0.97,-0.95,3).tolist():
    index = index +1
    input_s[:,0] = torch.full(input_s[:,0].shape, j)
    return_list = pinn_model_eval(input_s)
    result_list = return_list[0]
    title_list = return_list[1]
    print(len(title_list),len(result_list))

    analytic_x = result_list[0]
    analytic_y = result_list[1]
    analytic = (analytic_x**2 + analytic_y**2)**(0.5)
    #im = plt.scatter(input_s[:, 1].detach(), input_s[:, 2].detach(), c=analytic, cmap="jet")
    #plt.title("analytic")
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.colorbar(im)
    #plt.show()

    ta = input_s[:, 0]
    xa = input_s[:, 1]
    ya = input_s[:, 2]

    ta = ta.unsqueeze(-1)
    xa = xa.unsqueeze(-1)
    ya = ya.unsqueeze(-1)
    #t_decay = torch.sigmoid(5.0 * (2.0 - ((ta + 1.25) * 7.0)))
    #mean2_time = mean2 * t_decay
    #dist2 = MultivariateNormal(mean2_time.to(device), covariance_matrix2.to(device))

    point2 = torch.cat((xa, ya), dim=-1)
    #pdf2 = torch.exp(dist2.log_prob(point2))
    #ta = ta.squeeze()
    #t_activaton = torch.sigmoid(5.0 * (2.0 - ((ta + 1.25) * 7.0)))
    #analytic_activation_start2 = pdf2 #* t_activaton

    #dim = mean2.shape[0]
    #cov_det = torch.det(covariance_matrix2)
    #cov_inv = torch.inverse(covariance_matrix2)
    #norm_const = 1. / (torch.sqrt((2 * torch.pi) ** dim * cov_det))
    #x_sub_mean = point2 - mean2
    #exponent = -0.5 * torch.matmul(torch.matmul(x_sub_mean, cov_inv), x_sub_mean.t())
    #pdf2  = norm_const * torch.exp(exponent)

    #pdf2 = normalized_gaussian(point2, mean2,covariance_matrix2,ta)

    # Apply time decay
    ta = ta.squeeze()
    #t_decay = torch.sigmoid(5.0 * (2.0 - ((ta + 1.25) * 7.0)))
    #pdf2_time = pdf2 * t_decay
    #analytic_activation_start2 = pdf2_time#pdf2 #* t_activaton

    analytic_activation_start2 = modified_radial_basis_function2(point2, mean2,0, ta, decay_rate=1, rbf_sigma=1.0)



    #print(analytic_activation_start.shape)

    #im = plt.scatter(xa.detach(), ya.detach(), c=analytic_activation_start2, cmap="jet", vmin=0, vmax=1, s=0.05)
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.title(i)
   # plt.colorbar(im)
    #plt.show()

    for i in range(len(result_list)):
        value = result_list[i] *analytic_activation_start2
        title = title_list[i]

        if title=="analytic_y" or title=="analytic_x":
            im = plt.scatter(input_s[:,1].detach(), input_s[:,2].detach(), c=value, cmap="jet",s=0.05)
            plt.title(title)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.colorbar(im)
            plt.title("time = {}".format(j))
            plt.savefig("images/analytic/time={}.png".format(index))
            plt.show()




