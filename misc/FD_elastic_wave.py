import numpy as np
import matplotlib.pyplot as plt

def initial_condition_double_gaussian_derivative(sigma, mu_quake,nx, nz, dx, dz):
    # Create grid
    X, Z = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, nz))

    gauss = np.exp(- (X**2 + Z**2) / (2 * sigma**2))
    print("Position of Gaussian maximum:", np.unravel_index(np.argmax(gauss), gauss.shape))

    grad_x = -2 * X * gauss / (2 * sigma**2)
    grad_z = -2 * Z * gauss / (2 * sigma**2)

    u0x = -grad_x / np.max(np.abs(grad_x))
    u0y = grad_z / np.max(np.abs(grad_z))
    print("max init value = ",np.max(u0x))
    return u0x, u0y

def plot_initial_conditions(u0x, u0y, dx, dz):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    im1 = axs[0].imshow(u0x, extent=(-1, 1, -1, 1), origin="upper", aspect="auto", cmap="seismic")
    axs[0].set_title("u0x")
    plt.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(u0y, extent=(-1, 1, -1, 1), origin="upper", aspect="auto", cmap="seismic")
    axs[1].set_title("u0y")
    plt.colorbar(im2, ax=axs[1])

    plt.show()

def print_summary(nz, nx, dz, dx, vp, vs, rho, t_total, dt, nt, CFL):
    print("#################################################")
    print("2D elastic FDTD wave propagation in isotropic")
    print("medium in displacement formulation with")
    print("Cerjan(1985) boundary conditions")
    print("#################################################")
    print(f"Model:\n\t{nz} x {nx}\tgrid nz x nx\n\t{dz:.1e} x {dx:.1e}\t[m] dz x dx")
    print(f"\t{nx*dx:.1e} x {nz*dz:.1e}\t[m] model size")
    print(f"\t{np.min(vp):.1e} - {np.max(vp):.1e}\t[m/s] Vp min - max")
    print(f"\t{vs:.1e}\t[m/s] Vs")
    print(f"\t{np.min(rho):.1e} - {np.max(rho):.1e}\t[kg/m^3] rho min - max")
    print(f"\t{t_total:.1e}\t[s] total time")
    print(f"\t{dt:.1e}\t[s] dt\n\t{nt}\t[-] nt")
    print(f"\t{CFL:.2f}\t[-] Courant number")

# Output periodicity in time steps
IT_DISPLAY = 10

# MODEL
# Model dimensions, [m]
nx, nz = 401, 401
dx, dz = float(2/nx), float(2/nx)

# Elastic parameters
vp =2.0 * 0.89 * np.ones((nz, nx))
vs = 0.54
rho = 100.0 * np.ones(vp.shape)

# Lame parameters
lam = rho * (vp**2 - 2*vs**2)
mu = rho * vs**2

# TIME STEPPING
t_total = 1.0
dt = 0.8 / (np.max(vp) * np.sqrt(1.0/dx**2 + 1.0/dz**2))
nt = int(t_total/dt)
t = np.arange(nt + 1) * dt

CFL = np.max(vp) * dt * np.sqrt(1.0/dx**2 + 1.0/dz**2)
print("CFL = ",CFL)
sigma = 10.0 * min(dx, dz)
mu_quake = [0,0]
print(mu_quake)
print(nx,nz)


# Initialize Gaussian explosion as the initial condition
u0x, u0y = initial_condition_double_gaussian_derivative(sigma,mu_quake, nx, nz, dx, dz)

plot_initial_conditions(u0x, u0y, dx, dz)


# ABSORBING BOUNDARY
abs_thick = int(min(0.15 * nx, 0.15 * nz))
abs_rate = 0.1 / abs_thick

lmargin = [abs_thick, abs_thick]
rmargin = [abs_thick, abs_thick]
weights = np.ones((nz + 2, nx + 2))
for iz in range(nz + 2):
    for ix in range(nx + 2):
        i, j, k = 0, 0, 0
        if ix < lmargin[0] + 1:
            i = lmargin[0] + 1 - ix
        if iz < lmargin[1] + 1:
            k = lmargin[1] + 1 - iz
        if nx - rmargin[0] < ix:
            i = ix - nx + rmargin[0]
        if nz - rmargin[1] < iz:
            k = iz - nz + rmargin[1]
        if i == j == k == 0:
            continue
        rr = abs_rate**2 * (i**2 + j**2 + k**2)
        weights[iz, ix] = np.exp(-rr)

print(np.max(weights))

print_summary(nz, nx, dz, dx, vp, vs, rho, t_total, dt, nt, CFL)

# ALLOCATE MEMORY FOR WAVEFIELD
ux3 = np.zeros((nz + 2, nx + 2))
uz3 = np.zeros((nz + 2, nx + 2))
ux2 = np.zeros((nz + 2, nx + 2))
uz2 = np.zeros((nz + 2, nx + 2))
ux1 = np.zeros((nz + 2, nx + 2))
uz1 = np.zeros((nz + 2, nx + 2))

# Coefficients for derivatives
co_dxx = 1 / dx**2
co_dzz = 1 / dz**2
co_dxz = 1 / (4.0 * dx * dz)
co_dzx = 1 / (4.0 * dx * dz)
dt2rho = dt**2 / rho
lam_2mu = lam + 2 * mu

# Initialize fields
ux2[1:-1, 1:-1] = u0x
uz2[1:-1, 1:-1] = u0y

# Time loop
for it in range(nt):
    # Wavefield computations

    # Calculate spatial derivatives of the wavefield
    dux_dxx = co_dxx * (ux2[1:-1, :-2] - 2 * ux2[1:-1, 1:-1] + ux2[1:-1, 2:])[1:-1, 1:-1]
    dux_dzz = co_dzz * (ux2[:-2, 1:-1] - 2 * ux2[1:-1, 1:-1] + ux2[2:, 1:-1])[1:-1, 1:-1]
    dux_dxz = co_dxz * (ux2[:-2, 2:] - ux2[2:, 2:] - ux2[:-2, :-2] + ux2[2:, :-2])[1:-1, 1:-1]

    duz_dxx = co_dxx * (uz2[1:-1, :-2] - 2 * uz2[1:-1, 1:-1] + uz2[1:-1, 2:])[1:-1, 1:-1]
    duz_dzz = co_dzz * (uz2[:-2, 1:-1] - 2 * uz2[1:-1, 1:-1] + uz2[2:, 1:-1])[1:-1, 1:-1]
    duz_dzx = co_dzx * (uz2[:-2, 2:] - uz2[2:, 2:] - uz2[:-2, :-2] + uz2[2:, :-2])[1:-1, 1:-1]



    # Stress computations
    sigmas_ux = (lam_2mu[1:-1, 1:-1] * dux_dxx + lam[1:-1, 1:-1] * duz_dzx + mu[1:-1, 1:-1] * (dux_dzz + duz_dzx))
    sigmas_uz = (mu[1:-1, 1:-1] * (dux_dxz + duz_dxx) + lam[1:-1, 1:-1] * dux_dxz + lam_2mu[1:-1, 1:-1] * duz_dzz)

    # Displacement update using the stresses
    ux3[2:-2, 2:-2] = 2 * ux2[2:-2, 2:-2] - ux1[2:-2, 2:-2] + sigmas_ux * dt2rho[1:-1, 1:-1]
    uz3[2:-2, 2:-2] = 2 * uz2[2:-2, 2:-2] - uz1[2:-2, 2:-2] + sigmas_uz * dt2rho[1:-1, 1:-1]
    print("max value = ", np.max(ux3))


    # Apply absorbing boundary condition
    ux3 *= weights
    uz3 *= weights

    # Rotate arrays to shift in time
    ux1, ux2, ux3 = ux2, ux3, ux1
    uz1, uz2, uz3 = uz2, uz3, uz1

    # Displaying the wavefield at certain intervals
    if it % IT_DISPLAY == 0:
        plt.imshow(ux2[1:-1, 1:-1])
        plt.title(f'Time step: {it}')
        plt.colorbar()
        plt.pause(0.01)
        plt.clf()

# Final plot
plt.imshow(ux2[1:-1, 1:-1])
plt.title('Final Time Step')
plt.colorbar()
plt.show()
