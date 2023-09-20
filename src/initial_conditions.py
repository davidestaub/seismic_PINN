import configparser

import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(2)
import torch
import os


torch.set_printoptions(threshold=10000)
torch.set_printoptions(linewidth=200)

config = configparser.ConfigParser()
config.read("config.ini")


torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

#Parameters
lamda_solid = float(config['parameters']['lambda_solid'])
mu_solid = float(config['parameters']['mu_solid'])
rho_solid = float(config['parameters']['rho_solid'])
c2 = float(config['parameters']['c2'])
mu_quake_x = float(config['parameters']['mu_quake_x'])
mu_quake_y = float(config['parameters']['mu_quake_y'])
mu_quake = [mu_quake_x,mu_quake_y]
mu_quake = torch.tensor(mu_quake)
mu_quake = mu_quake.to(device)
mu_quake_x1 = float(config['parameters']['mu_quake_x1'])
mu_quake_y1 = float(config['parameters']['mu_quake_y1'])
mu_quake1 = [mu_quake_x1,mu_quake_y1]
mu_quake1 = torch.tensor(mu_quake1)
mu_quake1 = mu_quake1.to(device)
mu_quake_x2 = float(config['parameters']['mu_quake_x2'])
mu_quake_y2 = float(config['parameters']['mu_quake_y2'])
mu_quake2 = [mu_quake_x2,mu_quake_y2]
mu_quake2 = torch.tensor(mu_quake2)
mu_quake2 = mu_quake2.to(device)
sigma_quake = float(config['parameters']['sigma_quake'])
radius = float(config['parameters']['radius'])
T= float(config['parameters']['T'])
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
beta =beta.to(device)

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

   # print(t_initial.shape,r_abs.shape,alpha.shape)

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

def initial_condition_explosion(input_tensor, sigma):
    x = input_tensor[:, 1] - mu_quake[0]
    y = input_tensor[:, 2] - mu_quake[1]
    gauss = torch.exp(- (x**2 + y**2) / (2*sigma**2))
    grad_x = -2 * x * gauss / (2*sigma**2)
    grad_y = -2 * y * gauss / (2*sigma**2)
    u0x = -grad_x / torch.max(torch.abs(grad_x))
    u0y = -grad_y / torch.max(torch.abs(grad_y))
    return u0x, u0y


def initial_condition_explosion_two_sources(input_tensor, sigma):
    # Source 1
    x1 = input_tensor[:, 1] - mu_quake1[0]
    y1 = input_tensor[:, 2] - mu_quake1[1]
    gauss1 = torch.exp(- (x1 ** 2 + y1 ** 2) / (2 * sigma ** 2))
    grad_x1 = -2 * x1 * gauss1 / (2 * sigma ** 2)
    grad_y1 = -2 * y1 * gauss1 / (2 * sigma ** 2)

    # Source 2
    x2 = input_tensor[:, 1] - mu_quake2[0]
    y2 = input_tensor[:, 2] - mu_quake2[1]
    gauss2 = torch.exp(- (x2 ** 2 + y2 ** 2) / (2 * sigma ** 2))
    grad_x2 = -2 * x2 * gauss2 / (2 * sigma ** 2)
    grad_y2 = -2 * y2 * gauss2 / (2 * sigma ** 2)

    # Superposition of sources
    gauss = gauss1 + gauss2
    grad_x = grad_x1 + grad_x2
    grad_y = grad_y1 + grad_y2

    # Normalize gradients to get initial velocity field
    u0x = -grad_x / torch.max(torch.abs(grad_x))
    u0y = -grad_y / torch.max(torch.abs(grad_y))

    return u0x, u0y

def initial_condition_explosion_conditioned(input_tensor, sigma):
    x = input_tensor[:, 1] - input_tensor[:,3]
    y = input_tensor[:, 2] - input_tensor[:,4]

    # Generate 2D Gaussian distribution
    gauss = torch.exp(- (x**2 + y**2) / (2*sigma**2))

    # Compute gradients of the Gaussian
    grad_x = -2 * x * gauss / (2*sigma**2)
    grad_y = -2 * y * gauss / (2*sigma**2)

    # Normalize gradients to get initial velocity field
    #TODO check for discontinuitys
    u0x = -grad_x / torch.max(torch.abs(grad_x))
    u0y = -grad_y / torch.max(torch.abs(grad_y))

    return u0x, u0y

def initial_condition_explosion_conditioned_relative(input_tensor, sigma):
    print("inside correct function")

    #Input already centrized
    x = input_tensor[:, 1]
    y = input_tensor[:, 2]

    # Generate 2D Gaussian distribution
    gauss = torch.exp(- (x**2 + y**2) / (2*sigma**2))

    # Compute gradients of the Gaussian
    grad_x = -2 * x * gauss / (2*sigma**2)
    grad_y = -2 * y * gauss / (2*sigma**2)

    # Normalize gradients to get initial velocity field
    u0x = -grad_x / torch.max(torch.abs(grad_x))
    u0y = -grad_y / torch.max(torch.abs(grad_y))

    return u0x, u0y

def initial_condition_explosion_altered(input_tensor, sigma):
    r_smooth = 0.05
    # Create grid
    X = input_tensor[:, 1]
    Y = input_tensor[:, 2]
    R = torch.sqrt(X ** 2 + Y ** 2 + 1e-5)

    # Gaussian function
    f = torch.exp(-R ** 2 / (2 * sigma ** 2))

    # Raw g_x and g_y
    g_x_raw = X / R
    g_y_raw = Y / R

    # Weight function to smooth out the discontinuity
    weight = torch.clamp(R / r_smooth, max=1)

    # Weighted g_x and g_y
    g_x = weight * g_x_raw
    g_y = weight * g_y_raw

    # Now, compute the initial conditions
    u0x = f * g_x
    u0y = f * g_y

    return u0x, u0y

def initial_condition_donut(input_tensor, sigma):
    inner_radius = 0.1
    transition_width = 0.05
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

def initial_condition_gaussian(input_tensor,sigma):
    x_part = torch.pow(input_tensor[:, 1] - mu_quake[0], 2)
    y_part = torch.pow(input_tensor[:, 2] - mu_quake[1], 2)

    exponent = -0.5 * torch.pow((torch.sqrt(x_part + y_part + 1e-8) / sigma_quake), 2)
    earthquake_spike = torch.exp(exponent)
    u0x = earthquake_spike  # * solid_mask
    u0y = earthquake_spike  # * solid_mask

    return u0x,u0y