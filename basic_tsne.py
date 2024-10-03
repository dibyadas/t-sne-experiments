import os
import numpy as np
import matplotlib.pyplot as plt

# generate two oppositely inverted quadratic surface

# xy = np.random.uniform(-1, 1, (2,50))
# z = xy[0]**2 + xy[1]**2
# cl1 = np.r_[xy, z.reshape(1, -1)].T
# cl2 = np.r_[xy, -2 -1*z.reshape(1, -1)].T

# inp = np.r_[cl1, cl2]
# assert inp.shape == (100, 3,)



# Create the first criss-crossing dataset (a helix)
t1 = np.linspace(0, 4 * np.pi, 100)  # 100 points from 0 to 4π
x1 = np.sin(t1)
y1 = np.cos(t1)
z1 = t1
# Create the second criss-crossing dataset (another helix in opposite phase)
t2 = np.linspace(0, 4 * np.pi, 100)
x2 = np.sin(t2 + np.pi)  # Opposite phase
y2 = np.cos(t2 + np.pi)
z2 = t2

cl1 = np.c_[x1, y1, z2]
labels1 = np.ones(len(cl1))
cl2 = np.c_[x2, y2, z2]
labels2 = np.ones(len(cl2)) + 1
inp = np.r_[cl1, cl2]
labels = np.r_[labels1, labels2]

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print(beta)
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


# def compute_pij(X):

#     beta = np.ones((len(X), 1))
#     sum_X = np.sum(np.square(X), 1)
#     # get (xi - xj)^2, NxN matrix
#     # tiled_X = np.tile(X, (len(X),1))
#     # D = np.square(tiled_X - tiled_X.T)
#     D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
#     P = np.exp(-D.copy() * beta)
#     # sumP = sum(P)
#     # P = P / sumP
#     P[range(len(X)), range(len(X))] = 0
#     P = np.divide(P, np.tile(np.sum(P, axis=1), (len(X), 1)).T)
#     P = P + np.transpose(P)
#     P = P / np.sum(P)
#     P = np.maximum(P, 1e-12)
#     return P


# P = compute_pij(inp)
perplexity = 20
P = x2p(inp, 1e-5, perplexity)
P = P + np.transpose(P)
P = P / np.sum(P)
# P = P * 4.									# early exaggeration
P = np.maximum(P, 1e-12)


def save_plot(plot_Y, iter):
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title(f'Step: {iter}')
    plt.scatter(plot_Y[:, 0], plot_Y[:, 1], c=labels)
    plt.savefig(f'{output_folder_path}/{iter}.png')
    plt.close()
############################## numpy ############################

# max_iter = 100
# (n, d) = inp.shape
# no_dims = 2
# initial_momentum = 0.5
# final_momentum = 0.8
# eta = 100
# min_gain = 0.01
# # Y = np.random.randn(n, no_dims)
# Y = np.random.multivariate_normal([0,0], cov=[[0.0001,0], [0,0.0001]], size=(n,))
# dY = np.zeros((n, no_dims))
# iY = np.zeros((n, no_dims))
# gains = np.ones((n, no_dims))

# for iter in range(max_iter):
#     # Compute pairwise affinities
#     # tiled_Y = np.tile(Y, (len(Y),1))
#     # Dy = np.square(tiled_Y - tiled_Y.T)
#     # Q = 1. / (1. + Dy)
#     sum_Y = np.sum(np.square(Y), 1)
#     num = -2. * np.dot(Y, Y.T)
#     num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
#     num[range(n), range(n)] = 0.
#     Q = num / np.sum(num)
#     # Q = Q / np.sum(Q)
#     Q = np.maximum(Q, 1e-12)

#     # Compute gradient
#     factor = 4
#     PQ = P - Q
#     for i in range(n):
#         dY[i, :] = factor*np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

#     # Perform the update
#     if iter < 20:
#         momentum = initial_momentum
#     else:
#         momentum = final_momentum
#     gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
#             (gains * 0.8) * ((dY > 0.) == (iY > 0.))
#     gains[gains < min_gain] = min_gain
#     iY = momentum * iY - eta * (gains * dY)
#     Y = Y + iY
#     # Y = Y - np.tile(np.mean(Y, 0), (n, 1))

#     # Compute current value of cost function
#     if (iter + 1) % 10 == 0:
#         C = np.sum(P * np.log(P / Q))
#         print("Iteration %d: error is %f" % (iter + 1, C))

#     # Stop lying about P-values
#     if iter == 100:
#         factor = factor / 4.

###################################################################################


############################## torch without optimizer ############################

# import torch
# import torch.nn
# import torch.functional as F

# max_iter = 1000
# (n, d) = inp.shape
# no_dims = 2
# initial_momentum = 0.5
# final_momentum = 0.8
# eta = 500
# min_gain = 0.01
# # Y = np.random.randn(n, no_dims)
# # Y = np.random.multivariate_normal([0,0], cov=[[0.0001,0], [0,0.0001]], size=(n,))
# Y = torch.rand(n, no_dims)
# dY = torch.zeros((n, no_dims))
# iY = torch.zeros((n, no_dims))
# gains = torch.ones((n, no_dims))

# P = torch.from_numpy(P)



# for iter in range(max_iter):
#     # Compute pairwise affinities
#     # tiled_Y = np.tile(Y, (len(Y),1))
#     # Dy = np.square(tiled_Y - tiled_Y.T)
#     # Q = 1. / (1. + Dy)
#     sum_Y = torch.sum(torch.square(Y), 1)
#     num = -2. * torch.mm(Y, Y.T)
#     num = 1. / (1. + torch.add(torch.add(num, sum_Y).T, sum_Y))
#     num[range(n), range(n)] = 0.
#     Q = num / torch.sum(num)
#     # Q = Q / torch.sum(Q)
#     Q = torch.maximum(Q, torch.Tensor([1e-12]))

#     # Compute gradient
#     factor = 4
#     PQ = P - Q
#     for i in range(n):
#         dY[i, :] = factor*torch.sum(torch.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

#     # Perform the update
#     if iter < 20:
#         momentum = initial_momentum
#     else:
#         momentum = final_momentum
#     gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
#             (gains * 0.8) * ((dY > 0.) == (iY > 0.))
#     gains[gains < min_gain] = min_gain
#     iY = momentum * iY - eta * (gains * dY)
#     Y = Y + iY
#     # Y = Y - np.tile(np.mean(Y, 0), (n, 1))

#     # Compute current value of cost function
#     if (iter + 1) % 10 == 0:
#         C = torch.sum(P * torch.log(P / Q))
#         print("Iteration %d: error is %f" % (iter + 1, C))

#     # Stop lying about P-values
#     if iter == 100:
#         factor = factor / 4.


# plt.scatter(Y[:, 0], Y[:, 1])
# plt.show()

###################################################################################



# ############################## torch with optimizer ############################

# import torch
# import torch.nn as nn
# import torch.functional as F
# import torch.optim as optim

# max_iter = 2000
# (n, d) = inp.shape
# no_dims = 2
# initial_momentum = 0.5
# final_momentum = 0.8
# eta = 500
# min_gain = 0.01
# # Y = np.random.randn(n, no_dims)
# # Y = np.random.multivariate_normal([0,0], cov=[[0.0001,0], [0,0.0001]], size=(n,))
# # Y = torch.Tensor(Y)
# Y = torch.rand(n, no_dims)
# Y.requires_grad = True


# P = torch.from_numpy(P)
# P_initial_weight_mult_factor = 4.
# P = P*P_initial_weight_mult_factor

# # optimizer = optim.SGD([Y], lr=100, momentum=0.5)
# optimizer = optim.Adam([Y], lr=5, betas=(0.8, 0.5))

# output_folder_path = 'pics'
# if not os.path.exists(output_folder_path):
#     os.makedirs(output_folder_path)

# for iter in range(max_iter):
#     # Compute pairwise affinities
#     # tiled_Y = np.tile(Y, (len(Y),1))
#     # Dy = np.square(tiled_Y - tiled_Y.T)
#     # Q = 1. / (1. + Dy)
#     sum_Y = torch.sum(torch.square(Y), 1)
#     num = -2. * torch.mm(Y, Y.T)
#     num = 1. / (1. + torch.add(torch.add(num, sum_Y).T, sum_Y))
#     num[range(n), range(n)] = 0.
#     Q = num / torch.sum(num)
#     # Q = Q / torch.sum(Q)
#     Q = torch.maximum(Q, torch.Tensor([1e-12]))

#     optimizer.zero_grad()  #    
#     # Compute loss
#     loss = 1.0*torch.sum(P * torch.log(P / Q)) - 0.0*torch.sum((1-P) * torch.log((1-P) / (1-Q)))
    
#     # Backward pass (compute gradients)
#     loss.backward()
    
#     # Update the model parameters
#     optimizer.step()
    
#     # Print loss for monitoring
#     if iter % 10 == 0:
#         print(f'Epoch {iter}, Loss: {loss.item()}')

#     if iter == 100:
#         P = P / P_initial_weight_mult_factor

#     # if iter % 10 == 0:
        
#     #     plt.scatter(plot_Y[:, 0], plot_Y[:, 1], c=labels)
#     #     plt.savefig(f'{output_folder_path}/{iter}.png')
#     #     plt.close()

#     plot_Y = Y.clone().detach().numpy()
#     if iter <= 500:
#         # plot_Y = Y.clone().detach().numpy()
#         save_plot(plot_Y, iter)
        
#     if iter > 500 and iter % 10 == 0:
#         # plot_Y = Y.clone().detach().numpy()
#         save_plot(plot_Y, iter)


# Y = Y.detach().numpy()
# plt.scatter(Y[:, 0], Y[:, 1], c=labels)
# plt.show()

# ###################################################################################



############################## torch with mlp model and optimizer ############################

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

max_iter = 1000
(n, d) = inp.shape
no_dims = 2
initial_momentum = 0.5
final_momentum = 0.8
eta = 500
min_gain = 0.01
# Y = np.random.randn(n, no_dims)
# Y = np.random.multivariate_normal([0,0], cov=[[0.0001,0], [0,0.0001]], size=(n,))
# Y = torch.Tensor(Y)
Y = torch.rand(n, no_dims)
Y.requires_grad = True


P = torch.from_numpy(P)
# optimizer = optim.SGD([Y], lr=500, momentum=0.5)
# optimizer = optim.Adam([Y], lr=1e-5)

output_folder_path = 'pics'
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

def normalize_to_minus_one_to_one(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    
    # Normalize to [0, 1]
    tensor_normalized = (tensor - min_val) / (max_val - min_val)
    
    # Scale to [-1, 1]
    tensor_scaled = 2 * tensor_normalized - 1
    
    return tensor_scaled

num_points = len(inp)
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(3*num_points, num_points)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_points, 2*num_points)

    def forward(self, x: torch.Tensor):
        x = x.flatten()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view((num_points, 2))
        return x

# Example usage
import torch.nn.init as init

model = SimpleMLP().double()
def initialize_weights_gaussian(m):
    if isinstance(m, nn.Linear):
        # init.normal_(m.weight, mean=0.6, std=0.0)  # Gaussian init for weights
        init.normal_(m.weight, a=5, b=10)  # Gaussian init for weights
        if m.bias is not None:
            init.normal_(m.bias, mean=-1.0, std=0.02)  # Gaussian init for bias

# Apply the initialization to the model
# model.apply(initialize_weights_gaussian)
optimizer = optim.SGD(model.parameters(), lr=2, momentum=0.2)
# optimizer = optim.Adam(model.parameters(), lr=0.1)



noise = torch.randn(len(inp), 3)  # For example, shape (8, 3)
mean = 0
std_dev = 0.1
scaled_noise = mean + std_dev *noise
X = torch.from_numpy(inp) + scaled_noise
X = normalize_to_minus_one_to_one(X)

P_initial_weight_mult_factor = 1.
P = P*P_initial_weight_mult_factor

noise_schedule = { 100: 0.8, 500: 0.3, 1000: 0.3, 1500: 0.1, 2000: 0 }
std_dev = 1.2

for iter in range(max_iter):
    # Compute pairwise affinities
    # tiled_Y = np.tile(Y, (len(Y),1))
    # Dy = np.square(tiled_Y - tiled_Y.T)
    # Q = 1. / (1. + Dy)
    # X = (X - X.mean())
    # noise = torch.randn(len(inp), 2)  # For example, shape (8, 3)
    mean = 0
    try:
        std_dev = noise_schedule[iter]
    except KeyError:
        pass        

    scaled_noise = mean + std_dev * noise
    # scaled_noise = 0
    # X = torch.from_numpy(inp) + scaled_noise
    # X = normalize_to_minus_one_to_one(X)
    Y = model(X)
    # Y = Y + scaled_noise/10
    # Y = Y - torch.tile(torch.mean(Y, 0), (n, 1))
    sum_Y = torch.sum(torch.square(Y), 1)
    num = -2. * torch.mm(Y, Y.T)
    num = 1. / (1. + torch.add(torch.add(num, sum_Y).T, sum_Y))
    num[range(n), range(n)] = 0.
    Q = num / torch.sum(num)
    # Q = Q / torch.sum(Q)
    Q = torch.maximum(Q, torch.Tensor([1e-12]))

    optimizer.zero_grad()  #    
    # Compute loss
    loss = 1.0*torch.sum(P * torch.log(P / Q)) + 0.0*torch.sum(Q * torch.log(Q / P))
    # loss = 1.0*torch.sum(P * torch.log(P / Q)) - 1.0*torch.sum((1-P) * torch.log((1-P) / (1-Q)))
    
    # Backward pass (compute gradients)
    loss.backward()
    
    # Update the model parameters
    optimizer.step()
    
    # Print loss for monitoring
    if iter % 10 == 0:
        print(f'Epoch {iter}, Loss: {loss.item()}')
        # Print the gradient norm for each parameter
        # total_norm = 0.0
        # for param in model.parameters():
        #     param_norm = param.grad.data.norm(2)  # Compute L2 norm of the gradient
        #     total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # print(f'Epoch [{iter+1}/{max_iter}], Gradient Norm: {total_norm:.6f}')

    if iter == 100:
        P = P / P_initial_weight_mult_factor

    # if iter % 10 == 0:
    #     plot_Y = Y.clone().detach().numpy()
    #     plt.scatter(plot_Y[:, 0], plot_Y[:, 1], c=labels)
    #     plt.savefig(f'{output_folder_path}/{iter}.png')
    #     plt.close()

    plot_Y = Y.clone().detach().numpy()
    if iter <= 500:
        # plot_Y = Y.clone().detach().numpy()
        save_plot(plot_Y, iter)
        
    if iter > 500 and iter % 10 == 0:
        # plot_Y = Y.clone().detach().numpy()
        save_plot(plot_Y, iter)

Y = model(X)

Y = Y.detach().numpy()
# plt.scatter(Y[:, 0], Y[:, 1], c=labels)
# plt.show()

t1 = np.linspace(4 * np.pi, 8 * np.pi, 100)  # 100 points from 0 to 4π
x1 = np.sin(t1)
y1 = np.cos(t1)
z1 = t1
# Create the second criss-crossing dataset (another helix in opposite phase)
t2 = np.linspace(4 * np.pi, 8 * np.pi, 100)
x2 = np.sin(t2 + np.pi)  # Opposite phase
y2 = np.cos(t2 + np.pi)
z2 = t2

cl1 = np.c_[x1, y1, z2]
labels1 = np.ones(len(cl1)) + 2
cl2 = np.c_[x2, y2, z2]
labels2 = np.ones(len(cl2)) + 3
sinp = np.r_[cl1, cl2]
slabels = np.r_[labels1, labels2]
tX = normalize_to_minus_one_to_one(torch.from_numpy(sinp))
# tX = torch.from_numpy(tX)
tY = model(tX)
tY = tY.detach().numpy()

joined_map_pts = np.r_[Y, tY]
joined_labels = np.r_[labels, slabels]

plt.scatter(joined_map_pts[:, 0], joined_map_pts[:, 1], c=joined_labels)
plt.show()

plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title(f'Mapping of unseen data')
plt.scatter(joined_map_pts[:, 0], joined_map_pts[:, 1], c=joined_labels)
plt.savefig(f'torch_with_mlp_opt.png')
plt.close()

###################################################################################


# import torch
# import torch.nn as nn

# class SimpleMLP(nn.Module):`  `
#     def __init__(self):
#         super(SimpleMLP, self).__init__()
#         self.fc1 = nn.Linear(3, 64)  # Input size 3, output size 64
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(64, 32) # Hidden layer size 64 to 32
#         self.fc3 = nn.Linear(32, 2)  # Output size 2

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return x

# # Example usage
# model = SimpleMLP()
# input_data = torch.randn(8, 3)  # Example input with batch_size=8
# output = model(input_data)
# print(output)
