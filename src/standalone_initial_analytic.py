import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(2)
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

torch.set_printoptions(threshold=10000)
torch.set_printoptions(linewidth=200)

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



domain_extrema = torch.tensor([[-0.95, 0.0],  # Time dimension
                                    [-1.0, 1.0], [-1.0, 1.0]])  # Space dimension

soboleng = torch.quasirandom.SobolEngine(dimension=domain_extrema.shape[0])

def convert(tens):
    assert (tens.shape[1] == domain_extrema.shape[0])
    return tens * (domain_extrema[:, 1] - domain_extrema[:, 0]) + domain_extrema[:, 0]

n_collocation_points = 40000
input_s = convert(soboleng.draw(int(n_collocation_points)))
sorted_indices = torch.argsort(input_s[:, 0])
input_s = input_s[sorted_indices]
input_s.requires_grad = True


mu_quake = [0, 0]
sigma_quake = min(2, 1) * 0.12
radius = 0.2


lamda_solid = torch.tensor(2.0)#2.0 * 1e+8
mu_solid = torch.tensor(3.0)#3.0 * 1e+8
rho_solid = torch.tensor(1.0)#1000.0
torch.pi = torch.acos(torch.zeros(1)).item() * 2
theta = torch.tensor(0.25 * torch.pi)
# p wave speed
alpha = torch.tensor(torch.sqrt((lamda_solid + 2.0 * mu_solid) / rho_solid))
# s wave speed
beta = torch.sqrt(mu_solid / rho_solid)
T=torch.tensor(0.02)
M0 = torch.tensor(0.1)


def analytic_initial(input_tensor):
    offset=0.00

    t = torch.full([40000],-0.9)
    print(t.shape)
    x = input_tensor[: ,1]
    y = input_tensor[: ,2]

    r_abs = torch.sqrt(torch.pow((x - mu_quake[0]) ,2) + torch.pow((y - mu_quake[1]) ,2)+ 1e-5)
    r_abs = r_abs + 1e-5
    r_hat_x = (x - mu_quake[0] ) /r_abs
    r_hat_y = (y - mu_quake[1] ) /r_abs
    phi_hat_x = -1.0 * r_hat_y
    phi_hat_y = r_hat_x
    phi = torch.atan2(y-mu_quake[1],(x - mu_quake[0])+ 1e-5)
    M0_dot_input1 = (t + 1 + offset) - r_abs / alpha
    M0_dot_input2 = (t + 1 + offset) - r_abs / beta


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
    print(analytic_x)

    return analytic_x,analytic_y

ux1,uy1 = analytic_initial(input_s)
print(ux1.shape,uy1.shape)

im_init_dx = plt.scatter(input_s[:,1].detach().numpy(),input_s[:,2].detach().numpy(),c=ux1.detach().numpy(),s=5)
plt.colorbar(im_init_dx)
plt.title("analytic x")
plt.plot(1,1, marker="o")
plt.show()
im_init_dy = plt.scatter(input_s[:,1].detach().numpy(),input_s[:,2].detach().numpy(),c=uy1.detach().numpy(),s=5)
plt.colorbar(im_init_dy)
plt.title("analytic y")
plt.plot(1,1, marker="o")
plt.plot(0,1, marker="x")
plt.show()



def get_solid_residual(input_s):
    u_x,u_y = analytic_initial(input_s)
    u_x = u_x.unsqueeze(1)
    u_y = u_y.unsqueeze(1)
    gradient_x = torch.autograd.grad(u_x.sum(), input_s, create_graph=True)[0]
    print(gradient_x.shape)
    gradient_y = torch.autograd.grad(u_y.sum(), input_s, create_graph=True)[0]
    dt_x = gradient_x[:, 0]
    dx_x = gradient_x[:, 1]
    dy_x = gradient_x[:, 2]
    dt_y = gradient_y[:, 0]
    dx_y = gradient_y[:, 1]
    dy_y = gradient_y[:, 2]

    im_gradx_x = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dx_x.detach().numpy(), s=5)
    plt.colorbar(im_gradx_x)
    plt.title("dux/dx")
    plt.show()

    im_gradx_y = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dy_x.detach().numpy(),s=5)
    plt.colorbar(im_gradx_y)
    plt.title("dux/dy")
    plt.show()

    im_gradx_t = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dt_x.detach().numpy(),s=5)
    plt.colorbar(im_gradx_t)
    plt.title("dux/dt")
    plt.show()


    im_grady_x = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dx_y.detach().numpy(),s=5)
    plt.colorbar(im_grady_x)
    plt.title("duy/dx")
    plt.show()

    im_grady_y = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dy_y.detach().numpy(),s=5)
    plt.colorbar(im_grady_y)
    plt.title("duy/dy")
    plt.show()

    im_grady_t = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dt_y.detach().numpy(),s=5)
    plt.colorbar(im_grady_t)
    plt.title("duy/dt")
    plt.show()

    print(dt_x.requires_grad)
    dt2_x = torch.autograd.grad(dt_x.sum(), input_s, create_graph=True)[0][:, 0]
    dt2_y = torch.autograd.grad(dt_y.sum(), input_s, create_graph=True)[0][:, 0]
    print("dt2_x shape = ",dt2_x.shape)

    im_grady_t2 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dt2_y.detach().numpy(),s=5)
    plt.colorbar(im_grady_t2)
    plt.title("duy/dt2")
    plt.show()

    im_gradx_t2 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=dt2_x.detach().numpy(),s=5)
    plt.colorbar(im_gradx_t2)
    plt.title("dux/dt2")
    plt.show()


    # Reshape the gradients into tensors of shape [batch_size, 1]
    #dx_x = dx_x.view(-1, 1)
    #dy_x = dy_x.view(-1, 1)
    #dx_y = dx_y.view(-1, 1)
    #dy_y = dy_y.view(-1, 1)
    print(" dx_x shape = ",dx_x.shape)
    diag_1 = 2.0 * dx_x
    diag_2 = 2.0 * dy_y
    off_diag = dy_x + dx_y
    # Stack your tensors to a 2x2 tensor
    # The size of b will be (n_points, 2, 2)
    eps = 0.5 * torch.stack((torch.stack((diag_1, off_diag)), torch.stack((off_diag, diag_2))), dim=1)
    #eps = eps.squeeze()
    print("eps shape = ",eps.shape)

    im_eps00 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=eps[0,0,:].detach().numpy(),s=5)
    plt.colorbar(im_eps00)
    plt.title("eps00")
    plt.show()
    im_eps01 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=eps[0, 1, :].detach().numpy(), s=5)
    plt.colorbar(im_eps01)
    plt.title("eps01")
    plt.show()
    im_eps11 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=eps[1, 1, :].detach().numpy(), s=5)
    plt.colorbar(im_eps11)
    plt.title("eps11")
    plt.show()
    im_eps10 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=eps[1, 0, :].detach().numpy(), s=5)
    plt.colorbar(im_eps10)
    plt.title("eps10")
    plt.show()

    stress_tensor_00 = lamda_solid * (eps[0, 0] + eps[1, 1]) + 2.0 * mu_solid * eps[0, 0]
    stress_tensor_off_diag = 2.0 * mu_solid * eps[0, 1]
    stress_tensor_11 = lamda_solid * (eps[0, 0] + eps[1, 1]) + 2.0 * mu_solid * eps[1, 1]
    stress_tensor = torch.stack((torch.stack((stress_tensor_00, stress_tensor_off_diag)),
                                 torch.stack((stress_tensor_off_diag, stress_tensor_11))), dim=1)

    print("stress tensor shape =",stress_tensor.shape)

    im_sigma00 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=stress_tensor[0,0,:].detach().numpy(),s=5)
    plt.colorbar(im_sigma00)
    plt.title("sigma00")
    plt.show()
    im_sigma01 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=stress_tensor[0, 1, :].detach().numpy(), s=5)
    plt.colorbar(im_sigma01)
    plt.title("sigma01")
    plt.show()
    im_sigma11 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=stress_tensor[1, 1, :].detach().numpy(), s=5)
    plt.colorbar(im_sigma11)
    plt.title("sigma11")
    plt.show()
    im_sigma10 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=stress_tensor[1, 0, :].detach().numpy(), s=5)
    plt.colorbar(im_sigma10)
    plt.title("sigma10")
    plt.show()

    # Compute divergence of the stress tensor
    div_stress = torch.zeros(2,input_s.size(0), dtype=torch.float32, device=input_s.device)
    div_stress[0, :] = torch.autograd.grad(stress_tensor[0, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                       torch.autograd.grad(stress_tensor[0, 1].sum(), input_s, create_graph=True)[0][:, 2]
    div_stress[1, :] = torch.autograd.grad(stress_tensor[1, 0].sum(), input_s, create_graph=True)[0][:, 1] + \
                       torch.autograd.grad(stress_tensor[1, 1].sum(), input_s, create_graph=True)[0][:, 2]

    print("div stress shape =",div_stress.shape)

    im_div_stress0 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(), c=div_stress[0, :].detach().numpy(),s=5)
    plt.colorbar(im_div_stress0)
    plt.title("div_stress00")
    plt.show()
    im_div_stress1 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=div_stress[1, :].detach().numpy(), s=5)
    plt.colorbar(im_div_stress1)
    plt.title("div_stress01")
    plt.show()



    dt2_combined = torch.stack((dt2_x, dt2_y), dim=0)
    print("dt2_combined shape = ",dt2_combined.shape)
    residual_solid = rho_solid * dt2_combined - div_stress
    print("residual shape = ",residual_solid.shape)

    im_res0 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=residual_solid[0, :].detach().numpy(), s=5,vmin=torch.min(residual_solid[0, :]),vmax=torch.max(residual_solid[0, :]))
    plt.colorbar(im_res0)
    plt.title("res0")
    plt.show()
    im_res1 = plt.scatter(input_s[:, 1].detach().numpy(), input_s[:, 2].detach().numpy(),c=residual_solid[1, :].detach().numpy(), s=5,vmin=torch.min(residual_solid[1, :]),vmax=torch.max(residual_solid[1, :]))
    plt.colorbar(im_res1)
    plt.title("res1")
    plt.show()
    print(residual_solid)
    residual_solid = residual_solid.reshape(-1, )


    return residual_solid

residual = get_solid_residual(input_s)
print(residual)
residual_x = residual[:40000]
residual_y = residual[40000:]

im_res_x = plt.scatter(input_s[:,1].detach().numpy(),input_s[:,2].detach().numpy(),c=residual_x.detach().numpy(),s=5)
plt.colorbar(im_res_x)
plt.show()

im_res_y = plt.scatter(input_s[:,1].detach().numpy(),input_s[:,2].detach().numpy(),c=residual_y.detach().numpy(),s=5)
plt.colorbar(im_res_y)
plt.show()







