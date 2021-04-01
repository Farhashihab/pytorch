import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt

'''
1. Prepare Data
2. Create model
3. Loss and optimizer 
4. Traning loop
'''

# Let's prepare regression dataset

X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, random_state=1, noise=20)
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0], 1)

n_samples, n_features = X.shape

# Let's define the model
input_size = n_features
output_size = 1


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define diferent layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)


model = LinearRegression(input_size, output_size)

# loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epoch = 100

for epoch in range(num_epoch):
    # Farward pass and loss

    y_predicted = model(X)
    loss = criterion(y_predicted, Y)

    # Backward pass

    loss.backward()

    # Update weights
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:

        print(f"Epoch {epoch + 1}, loss : {loss.item():.4f}")

# Plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()

# print(X_numpy)
# print(Y_numpy)
