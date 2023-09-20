if config['test_loss']['on'] == 'True':
    numpoints_sqrt = 256
    dt = 1.1e-3
    mu_quake_x = 0.0
    mu_quake_y = 0.0

    mu_quake = [mu_quake_x, mu_quake_y]
    mu_quake = torch.tensor(mu_quake)
    pinn_domain_extent_x = [float(config["domain"]["xmin"]), float(config["domain"]["xmax"])]
    pinn_domain_extent_y = [float(config["domain"]["ymin"]), float(config["domain"]["ymax"])]
    pinn_length_x = pinn_domain_extent_x[1] - pinn_domain_extent_x[0]
    pinn_length_y = pinn_domain_extent_y[1] - pinn_domain_extent_y[0]

    enlarge_factor = 3
    devito_length_x = enlarge_factor * pinn_length_x
    devito_length_y = enlarge_factor * pinn_length_y
    extent = (devito_length_x, devito_length_y)
    devito_center = [devito_length_x / 2.0, devito_length_y / 2.0]

    mu_quake_devito = [devito_center[0] + mu_quake[0].numpy(), devito_center[1] + mu_quake[1].numpy()]

    # increase Devito domain to avoid Boundary issues (reflections)
    shape = (numpoints_sqrt * enlarge_factor, numpoints_sqrt * enlarge_factor)
    x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0] / (shape[0] - 1)))
    y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=extent[1] / (shape[1] - 1)))
    grid = Grid(extent=extent, shape=shape, dimensions=(x, y))
    spacing = (extent[0] / shape[0], extent[0] / shape[0])
    X, Y = np.meshgrid(np.linspace(0, numpoints_sqrt * spacing[0], numpoints_sqrt),
                       np.linspace(0, numpoints_sqrt * spacing[1], numpoints_sqrt))

    if model_type == "mixture":
        # Generate mixtures for mu and lambda
        mu_mixture = FD_devito.generate_mixture().numpy()
        lambda_mixture = FD_devito.generate_mixture().numpy()
        lambda_solid = FD_devito.compute_param_np(X, Y, lambda_mixture)
        mu_solid = FD_devito.compute_param_np(X, Y, mu_mixture)
    elif model_type == "constant":
        lambda_solid = np.full((X, Y), config["parameters"]["lambda_solid"])
        mu_solid = np.full((X, Y), config["parameters"]["mu_solid"])
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
    lambda_large[:, y_offset + numpoints_sqrt:] = lambda_large[:, y_offset + (numpoints_sqrt - 1)][:,
                                                  np.newaxis]

    lambda_solid = lambda_large

    Xp, Yp = np.meshgrid(
        np.linspace(0, (numpoints_sqrt * enlarge_factor) * spacing[0], numpoints_sqrt * enlarge_factor),
        np.linspace(0, (numpoints_sqrt * enlarge_factor) * spacing[1], numpoints_sqrt * enlarge_factor))
    plt.scatter(Xp, Yp, c=lambda_solid)
    plt.show()
    mu_large = np.full((X_large, Y_large), 0.0)
    mu_large[x_offset:x_offset + numpoints_sqrt, y_offset:y_offset + numpoints_sqrt] = mu_solid
    mu_large[0:x_offset, :] = mu_large[x_offset, :]
    mu_large[x_offset + numpoints_sqrt:, :] = mu_large[x_offset + (numpoints_sqrt - 1), :]
    mu_large[:, 0:y_offset] = mu_large[:, y_offset][:, np.newaxis]
    mu_large[:, y_offset + numpoints_sqrt:] = mu_large[:, y_offset + (numpoints_sqrt - 1)][:, np.newaxis]
    mu_solid = mu_large

    # Using both ux and uy as separate field for easier comparison to PINNs and easier pde formulation
    ux_devito = TimeFunction(name='ux_devito', grid=grid, space_order=4, time_order=2)
    uy_devito = TimeFunction(name='uy_devito', grid=grid, space_order=4, time_order=2)

    lambda_ = Function(name='lambda_f', grid=grid, space_order=4)
    mu_ = Function(name='mu_f', grid=grid, space_order=4)
    initialize_function(lambda_, lambda_solid, nbl=0)
    initialize_function(mu_, mu_solid, nbl=0)
    print(mu_.data, mu_.data.shape)
    plt.scatter(Xp, Yp, c=lambda_.data)
    plt.show()
    plt.scatter(Xp, Yp, c=mu_.data)
    plt.show()

    # The initial condition that also was used for The PINN, mimics a perfectly isotropic explosion
    u0x_devito, u0y_devito = FD_devito.initial_condition_double_gaussian_derivative(
        sigma=float(config["parameters"]["sigma_quake"]), mu_quake=mu_quake_devito, nx=shape[0],
        ny=shape[1], dx=spacing[0], dy=spacing[1])

    # Initialize the VectorTimeFunction with the initial values
    # Setting both u(t0,x,y) and u(t0+dt,x,y) to u0 to guarantee du0/dt=0
    # and because 2nd order time discretization is neeed
    ux_devito.data[0] = u0x_devito
    uy_devito.data[0] = u0y_devito
    ux_devito.data[1] = u0x_devito
    uy_devito.data[1] = u0y_devito

    FD_devito.plot_field(ux_devito.data[0])
    FD_devito.plot_field(uy_devito.data[0])
    # Plot from initial field look good
    print(ux_devito.shape, uy_devito.shape)

    print("start stress calculation")
    # Divergence of stress tensor for ux
    div_stress_ux = (
                                lambda_ + 2.0 * mu_) * ux_devito.dx2 + mu_ * ux_devito.dy2 + lambda_ * uy_devito.dy.dx + mu_ * uy_devito.dx.dy

    # Divergence of stress tensor for uy
    div_stress_uy = (
                                lambda_ + 2.0 * mu_) * uy_devito.dy2 + mu_ * uy_devito.dx2 + lambda_ * ux_devito.dx.dy + mu_ * ux_devito.dy.dx

    print("stress calculated")
    # Elastic wave equation for ux and uy
    # print("model damp = ",model.damp,model.damp.data)
    pde_x = rho_solid * ux_devito.dt2 - div_stress_ux
    pde_y = rho_solid * uy_devito.dt2 - div_stress_uy

    # BOundary conditions:
    x, y = grid.dimensions
    t = grid.stepping_dim
    ny, nx = shape[0], shape[1]
    bc = [Eq(ux_devito[t + 1, x, 0], 0.)]
    bc += [Eq(ux_devito[t + 1, x, ny - 1], 0.)]
    bc += [Eq(ux_devito[t + 1, 0, y], 0.)]
    bc += [Eq(ux_devito[t + 1, nx - 1, y], 0.)]

    bc += [Eq(ux_devito[t + 1, x, 1], 0.)]
    bc += [Eq(ux_devito[t + 1, x, ny - 2], 0.)]
    bc += [Eq(ux_devito[t + 1, 1, y], 0.)]
    bc += [Eq(ux_devito[t + 1, nx - 2, y], 0.)]

    bc += [Eq(uy_devito[t + 1, x, 0], 0.)]
    bc += [Eq(uy_devito[t + 1, x, ny - 1], 0.)]
    bc += [Eq(uy_devito[t + 1, 0, y], 0.)]
    bc += [Eq(uy_devito[t + 1, nx - 1, y], 0.)]

    bc += [Eq(uy_devito[t + 1, x, 1], 0.)]
    bc += [Eq(uy_devito[t + 1, x, ny - 2], 0.)]
    bc += [Eq(uy_devito[t + 1, 1, y], 0.)]
    bc += [Eq(uy_devito[t + 1, nx - 2, y], 0.)]

    # Formulating stencil to solve for u forward
    stencil_x = Eq(ux_devito.forward, solve(pde_x, ux_devito.forward))
    stencil_y = Eq(uy_devito.forward, solve(pde_y, uy_devito.forward))
    pprint(stencil_x)
    pprint(stencil_y)
    op = Operator([stencil_x] + [stencil_y] + bc)
    time = 0
    index = 0
    res_list_ux = []
    res_list_uy = []
    res_list_u = []
    res_list_devito_x = []
    res_list_devito_y = []
    res_list_devito_u = []
    time_list = np.linspace(0, 1, 101).tolist()
    devito_time = 0.0
    for i in time_list:
        res_list_devito_x.append(np.transpose(ux_devito.data[1][x_offset:x_offset + numpoints_sqrt,
                                              y_offset:y_offset + numpoints_sqrt]).copy())
        res_list_devito_y.append(np.transpose(uy_devito.data[1][x_offset:x_offset + numpoints_sqrt,
                                              y_offset:y_offset + numpoints_sqrt]).copy())
        res_list_devito_u.append(np.sqrt(np.transpose(uy_devito.data[1][x_offset:x_offset + numpoints_sqrt,
                                                      y_offset:y_offset + numpoints_sqrt]).copy() ** 2 + np.transpose(
            ux_devito.data[1][x_offset:x_offset + numpoints_sqrt,
            y_offset:y_offset + numpoints_sqrt]).copy() ** 2))

        op(time_M=10, dt=dt)
        time = i
        inputs = self.soboleng.draw(int(pow(numpoints_sqrt, 2)))
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1.0, 1.0, numpoints_sqrt),
                                        torch.linspace(-1.0, 1.0, numpoints_sqrt))
        grid_x = torch.reshape(grid_x, (-1,))
        grid_y = torch.reshape(grid_y, (-1,))

        inputs[:, 1] = grid_x
        inputs[:, 2] = grid_y
        inputs[:, 0] = time
        inputs[:, 3] = mu_quake_test[0]
        inputs[:, 4] = mu_quake_test[1]

        ux = self.pinn_model_eval(inputs)[:, 0]
        uy = self.pinn_model_eval(inputs)[:, 1]
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

    for h in range(0, len(res_list_uy)):
        diffx = ((res_uy[h, :, :]) - (res_list_devito_x[h]))
        diffx[0:9, :] = 0
        diffx[:, 0:9] = 0
        diffx[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
        diffx[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0

        diffy = (res_ux[h, :, :]) - (res_list_devito_y[h])
        diffy[0:9, :] = 0
        diffy[:, 0:9] = 0
        diffy[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
        diffy[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0

        diffu = ((res_u[h, :, :]) - (res_list_devito_u[h]))
        diffu[0:9, :] = 0
        diffu[:, 0:9] = 0
        diffu[:, numpoints_sqrt - 9:numpoints_sqrt] = 0
        diffu[numpoints_sqrt - 9:numpoints_sqrt:, :] = 0