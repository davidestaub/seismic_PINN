import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_mixture(num_mixtures=100, amplitude=0.3, width=0.15):
    np.random.seed(42)
    torch.manual_seed(42)
    return torch.tensor(np.concatenate([
        np.random.uniform(-1, 1, (num_mixtures, 2)), # location
        width * np.ones((num_mixtures, 1)), # width
        amplitude * np.random.uniform(10, 50, (num_mixtures, 1)), # amplitude
    ], axis=1), dtype=torch.float32)

def compute_param(X, Y, mixture):
    print("X.shape = ",X.shape)
    param_mixture = torch.exp(-0.5 * ((X.unsqueeze(-1) - mixture[:, 0])**2 + (Y.unsqueeze(-1) - mixture[:, 1])**2) / (mixture[:, 2]**2))
    param = torch.sum(mixture[:, 3] * param_mixture, dim=-1)
    return param

def compute_image(mixture, res=100):
    x_vals = torch.linspace(-1, 1, res)
    y_vals = torch.linspace(-1, 1, res)
    X, Y = torch.meshgrid(x_vals, y_vals)
    image = compute_param(X, Y, mixture)
    return image


def sigmoid(x, a=1, b=0):
    return 1 / (1 + torch.exp(-a * (x - b)))


def compute_lambda_mu_layers(X, Y, num_layers, smoothing_fraction=0.2):
    X = X.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    np.random.seed(42)
    torch.manual_seed(42)

    min_val = 10
    max_val = 50
    layer_thickness = X.shape[0] // num_layers
    smoothing_thickness = int(layer_thickness * smoothing_fraction)

    lambda_s = torch.zeros_like(X)
    mu_s = torch.zeros_like(Y)

    current_lambda_val = np.random.uniform(min_val, max_val)
    current_mu_val = np.random.uniform(min_val, max_val)

    for i in range(num_layers):
        next_lambda_val = np.random.uniform(min_val, max_val)
        next_mu_val = np.random.uniform(min_val, max_val)

        start_idx = i * layer_thickness
        end_idx = (i + 1) * layer_thickness if i != num_layers - 1 else X.shape[0]

        # Set the base value for this layer
        lambda_s[start_idx:end_idx, :] = current_lambda_val
        mu_s[start_idx:end_idx, :] = current_mu_val

        # Sigmoid smoothing at the upper boundary of this layer
        interp_start = end_idx - smoothing_thickness
        interp_end = end_idx

        x = torch.linspace(-5, 5, steps=interp_end - interp_start)
        transition = sigmoid(x)

        interpolated_lambda = (1 - transition) * current_lambda_val + transition * next_lambda_val
        interpolated_mu = (1 - transition) * current_mu_val + transition * next_mu_val

        # Make these interpolated values compatible for 2D assignment
        interpolated_lambda = interpolated_lambda[:, None].expand(-1, X.shape[1])
        interpolated_mu = interpolated_mu[:, None].expand(-1, X.shape[1])

        lambda_s[interp_start:interp_end, :] = interpolated_lambda
        mu_s[interp_start:interp_end, :] = interpolated_mu

        current_lambda_val = next_lambda_val
        current_mu_val = next_mu_val

    return lambda_s.squeeze(), mu_s.squeeze()


if __name__ == "__main__":
    x_vals = torch.linspace(-1, 1, 1000)
    y_vals = torch.linspace(-1, 1, 1000)
    X, Y = torch.meshgrid(x_vals, y_vals)
    lambda_l,mu_l = compute_lambda_mu_layers(X,Y,5)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    img1 = ax[0].imshow(mu_l, extent=[-1, 1, -1, 1], cmap='viridis')
    ax[0].set_title('Mu Layer Model')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    fig.colorbar(img1, ax=ax[0])

    img2 = ax[1].imshow(lambda_l, extent=[-1, 1, -1, 1], cmap='viridis')
    ax[1].set_title('Lambda Layere Model')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    fig.colorbar(img2, ax=ax[1])

    # Generate mixtures for mu and lambda
    mu_mixture = generate_mixture()
    lambda_mixture = generate_mixture()

    mu_image = compute_image(mu_mixture)
    lambda_image = compute_image(lambda_mixture)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    img1 = ax[0].imshow(mu_image, extent=[-1, 1, -1, 1], cmap='viridis')
    ax[0].set_title('Mu Mixture Model')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    fig.colorbar(img1, ax=ax[0])

    img2 = ax[1].imshow(lambda_image, extent=[-1, 1, -1, 1], cmap='viridis')
    ax[1].set_title('Lambda Mixture Model')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    fig.colorbar(img2, ax=ax[1])

    plt.show()