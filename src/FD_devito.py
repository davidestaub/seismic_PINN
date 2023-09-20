import numpy as np
from cached_property import cached_property
from devito import *
import matplotlib.pyplot as plt
from matplotlib import cm
from sympy import pprint
import torch
from devito.builtins import initialize_function


def generate_mixture(num_mixtures=100, amplitude=0.3, width=0.15):
    np.random.seed(42)
    torch.manual_seed(42)
    return torch.tensor(np.concatenate([
        np.random.uniform(0, 2, (num_mixtures, 2)), # location
        width * np.ones((num_mixtures, 1)), # width
        amplitude * np.random.uniform(10, 50, (num_mixtures, 1)), # amplitude
    ], axis=1), dtype=torch.float32)

# Generate mixtures for mu and lambda
mu_mixture = generate_mixture().numpy()
lambda_mixture = generate_mixture().numpy()


def compute_param_np(X, Y, mixture):
    param = np.zeros_like(X)  # Initialize the result to zeros of the same shape as X
    for mix in mixture:  # Loop through each mixture
        loc_x, loc_y, width, amplitude = mix
        param_mixture = np.exp(-0.5 * ((X - loc_x)**2 + (Y - loc_y)**2) / (width**2))
        param += amplitude * param_mixture  # Add this to the accumulated sum
    return param

def plot_field(field, xmin=0., xmax=2., ymin=0., ymax=2., zmin=None, zmax=None,
               view=None, linewidth=0):
    """
    Utility plotting routine for 2D data.

    Parameters
    ----------
    field : array_like
        Field data to plot.
    xmax : int, optional
        Length of the x-axis.
    ymax : int, optional
        Length of the y-axis.
    view: int, optional
        View point to intialise.
    """
    if xmin > xmax or ymin > ymax:
        raise ValueError("Dimension min cannot be larger than dimension max.")
    if (zmin is not None and zmax is not None):
        if zmin > zmax:
            raise ValueError("Dimension min cannot be larger than dimension max.")
    elif(zmin is None and zmax is not None):
        if np.min(field) >= zmax:
            warning("zmax is less than field's minima. Figure deceptive.")
    elif(zmin is not None and zmax is None):
        if np.max(field) <= zmin:
            warning("zmin is larger than field's maxima. Figure deceptive.")
    x_coord = np.linspace(xmin, xmax, field.shape[0])
    y_coord = np.linspace(ymin, ymax, field.shape[1])
    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x_coord, y_coord, indexing='ij')
    ax.plot_surface(X, Y, field[:], cmap=cm.viridis, rstride=1, cstride=1,
                    linewidth=linewidth, antialiased=False)

    # Enforce axis measures and set view if given
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if zmin is None:
        zmin = np.min(field)
    if zmax is None:
        zmax = np.max(field)
    ax.set_zlim(zmin, zmax)

    if view is not None:
        ax.view_init(*view)

    # Label axis
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    plt.show()


def initial_condition_double_gaussian_derivative(sigma, mu_quake, nx, ny, dx, dy):
    # Create grid
    X, Y = np.meshgrid(np.linspace(0, nx * dx, nx), np.linspace(0, ny * dy, ny))

    # Shifted Gaussian to center it at mu_quake
    print(type(X),type(mu_quake[0]),type(mu_quake))
    gauss = np.exp(- ((X - mu_quake[0]) ** 2 + (Y - mu_quake[1]) ** 2) / (2 * sigma ** 2))
    print("Position of Gaussian maximum:", np.unravel_index(np.argmax(gauss), gauss.shape))

    # Adjusted derivatives
    grad_x = -2 * (X - mu_quake[0]) * gauss / (2 * sigma ** 2)
    grad_z = -2 * (Y - mu_quake[1]) * gauss / (2 * sigma ** 2)

    u0x = -grad_x / np.max(np.abs(grad_x))
    u0y = -grad_z / np.max(np.abs(grad_z))
    print("max init value =", np.max(u0x))

    return u0y,u0x


class TimeAxis(object):
    """
    Data object to store the TimeAxis. Exactlz three of the four kez arguments
    must be prescribed. Because of remainder values, it is not possible to create
    a TimeAxis that exactlz adheres to the inputs; therefore, start, stop, step
    and num values should be taken from the TimeAxis object rather than relzing
    upon the input values.

    The four possible cases are:
    start is None: start = step*(1 - num) + stop
    step is None: step = (stop - start)/(num - 1)
    num is None: num = ceil((stop - start + step)/step);
                 because of remainder stop = step*(num - 1) + start
    stop is None: stop = step*(num - 1) + start

    Parameters
    ----------
    start : float, optional
        Start of time axis.
    step : float, optional
        Time interval.
    num : int, optional
        Number of values (Note: this is the number of intervals + 1).
        Stop value is reset to correct for remainder.
    stop : float, optional
        End time.
    """
    def __init__(self, start=None, step=None, num=None, stop=None):
        try:
            if start is None:
                start = step*(1 - num) + stop
            elif step is None:
                step = (stop - start)/(num - 1)
            elif num is None:
                num = int(np.ceil((stop - start + step)/step))
                stop = step*(num - 1) + start
            elif stop is None:
                stop = step*(num - 1) + start
            else:
                raise ValueError("Onlz three of start, step, num and stop maz be set")
        except:
            raise ValueError("Three of args start, step, num and stop maz be set")

        if not isinstance(num, int):
            raise TypeError("input argument must be of tzpe int")

        self.start = float(start)
        self.stop = float(stop)
        self.step = float(step)
        self.num = int(num)

    def __str__(self):
        return "TimeAxis: start=%g, stop=%g, step=%g, num=%g" % \
               (self.start, self.stop, self.step, self.num)

    def _rebuild(self):
        return TimeAxis(start=self.start, stop=self.stop, num=self.num)

    @cached_property
    def time_values(self):
        return np.linspace(self.start, self.stop, self.num)

if __name__ == "__main__":
    extent = (6., 6.)
    shape = (601, 601)
    x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))
    y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=extent[1]/(shape[1]-1)))
    grid = Grid(extent=extent, shape=shape, dimensions=(x,y))
    spacing = (extent[0]/shape[0],extent[0]/shape[0])
    rho_solid = 100.0
    X, Y = np.meshgrid(np.linspace(0, (201) * spacing[0], 201), np.linspace(0, (201) * spacing[1], 201))
    print(x.shape,mu_mixture.shape)
    lambda_solid = compute_param_np(X,Y, lambda_mixture)


    # Create a larger array (7x7)
    X_large, Y_large = 601, 601
    lambda_large = np.full((X_large, Y_large), 20.0)
    # Calculate the position to place the smaller array at the center of the larger array
    x_offset = (X_large - 201) // 2
    y_offset = (Y_large - 201) // 2

    # Place the smaller array in the center of the larger array
    lambda_large[x_offset:x_offset + 201, y_offset:y_offset + 201] = lambda_solid
    lambda_large[0:x_offset, :] = lambda_large[x_offset, :]
    lambda_large[x_offset + 201:, :] = lambda_large[x_offset + 200, :]
    lambda_large[:, 0:y_offset] = lambda_large[:, y_offset][:, np.newaxis]
    lambda_large[:, y_offset + 201:] = lambda_large[:, y_offset + 200][:, np.newaxis]

    lambda_solid = lambda_large

    Xp, Yp = np.meshgrid(np.linspace(0, (601) * spacing[0], 601), np.linspace(0, (601) * spacing[1], 601))
    print(type(lambda_solid),lambda_solid.shape)
    plt.scatter(Xp,Yp,c=lambda_solid)
    plt.show()
    print(lambda_solid.shape)
    mu_solid = compute_param_np(X,Y, mu_mixture)
    mu_large = np.full((X_large, Y_large), 30.0)
    mu_large[x_offset:x_offset + 201, y_offset:y_offset + 201] = mu_solid
    mu_large[0:x_offset, :] = mu_large[x_offset, :]
    mu_large[x_offset + 201:, :] = mu_large[x_offset + 200, :]
    mu_large[:, 0:y_offset] = mu_large[:, y_offset][:, np.newaxis]
    mu_large[:, y_offset + 201:] = mu_large[:, y_offset + 200][:, np.newaxis]
    mu_solid = mu_large

    #mu_solid = np.full((X_large, Y_large), 30.0)
    #lambda_solid = np.full((X_large, Y_large), 20.0)
    dt=1e-3
    time_range = TimeAxis(start=0.0, stop=1.0, step=dt)

    # = demo_model('constant-isotropic',
                       #shape=shape,  # Number of grid points.
                       #spacing=spacing,  # Grid spacing in m.
                       #nbl=20, space_order=4)      # boundary layer.
    #lambda_solid=20.0
    #mu_solid=30.0

    #Using both ux and uy as separate field for easier comparison to PINNs and easier pde formulation
    ux = TimeFunction(name='ux', grid=grid, space_order=4, time_order=2)
    uy = TimeFunction(name='uy', grid=grid, space_order=4, time_order=2)
    print(uy)
    print(ux.data.shape)

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
    u0x, u0y = initial_condition_double_gaussian_derivative(sigma=0.1, mu_quake=[3.0,3.0], nx=shape[0], ny=shape[1], dx=spacing[0], dy=spacing[1])

    # Initialize the VectorTimeFunction with the initial values
    # Setting both u(t0,x,y) and u(t0+dt,x,y) to u0 to guarantee du0/dt=0
    # and because 2nd order time discretization is neeed
    ux.data[0] = u0x
    uy.data[0] = u0y
    ux.data[1] = u0x
    uy.data[1] = u0y


    plot_field(ux.data[0])
    plot_field(uy.data[0])
    #Plot from initial field look good
    print(ux.shape,uy.shape)

    print("start stress calculation")
    # Divergence of stress tensor for ux
    div_stress_ux = (lambda_ + 2.0 * mu_) * ux.dx2 + mu_ * ux.dy2 + lambda_ * uy.dy.dx + mu_ * uy.dx.dy

    # Divergence of stress tensor for uy
    div_stress_uy = (lambda_ + 2.0 * mu_) * uy.dy2 + mu_ * uy.dx2 + lambda_ * ux.dx.dy + mu_ * ux.dy.dx

    print("stress calculated")
    # Elastic wave equation for ux and uy
    #print("model damp = ",model.damp,model.damp.data)
    pde_x = rho_solid * ux.dt2 - div_stress_ux #+model.damp * ux.dt
    pde_y = rho_solid * uy.dt2 - div_stress_uy #+model.damp * uy.dt


    #BOundary conditions:
    x, y = grid.dimensions
    t = grid.stepping_dim
    ny,nx = shape[0],shape[1]
    bc = [Eq(ux[t+1,x, 0], 0.)]
    bc += [Eq(ux[t+1,x, ny-1], 0.)]
    bc += [Eq(ux[t+1,0, y], 0.)]
    bc += [Eq(ux[t+1,nx-1, y], 0.)]

    bc += [Eq(ux[t+1,x, 1], 0.)]
    bc += [Eq(ux[t+1,x, ny-2], 0.)]
    bc += [Eq(ux[t+1,1, y], 0.)]
    bc += [Eq(ux[t+1,nx-2, y], 0.)]

    bc += [Eq(uy[t+1,x, 0], 0.)]
    bc += [Eq(uy[t+1,x, ny-1], 0.)]
    bc += [Eq(uy[t+1,0, y], 0.)]
    bc += [Eq(uy[t+1,nx-1, y], 0.)]

    bc += [Eq(uy[t+1,x, 1], 0.)]
    bc += [Eq(uy[t+1,x, ny-2], 0.)]
    bc += [Eq(uy[t+1,1, y], 0.)]
    bc += [Eq(uy[t+1,nx-2, y], 0.)]


    #Formulating stencil to solve for u forward
    stencil_x = Eq(ux.forward,solve(pde_x,ux.forward))
    stencil_y = Eq(uy.forward,solve(pde_y,uy.forward))
    pprint(stencil_x)
    pprint(stencil_y)
    op = Operator([stencil_x]+[stencil_y]+bc)

    time= 0
    index = 0
    print("start")
    while time <= 1.0:
        print(time)
        index = index +1
        plt.imshow(np.transpose(ux.data[0][x_offset:x_offset + 201, y_offset:y_offset + 201]),vmin=-0.2,vmax=0.2)
        plt.title("ux time={}".format(time))
        plt.show()
        plt.imshow(np.transpose(uy.data[0][x_offset:x_offset + 201, y_offset:y_offset + 201]), vmin=-0.2, vmax=0.2)
        plt.title("uy time={}".format(time))
        plt.show()
        for i in range(0,2):
            op(time=10, dt=dt)
        time = time + 10*dt





