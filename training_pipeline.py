'''
steps:
1. Design model (input_size,output_size,forward_pass)
2. constract the loss and optimizer
3. Training Loop
    - Forward Pass : Compute Prediction
    - backward pass : Gradients
    - Update weights
'''

import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

x_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape

input_dim = n_features
output_dim = n_features
print(n_samples, n_features)


# model = nn.Linear(n_features,n_features)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define diferent layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
    
model = LinearRegression(input_dim, output_dim)


# model = LinearRegression(input_dim, output_dim)

print(f" Prediction before training f(5) {model(x_test).item():.3f}")

# training
learning_rate = 0.01
n_iter = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_model(X, Y, iteration):
    for epoch in range(iteration):
        y_pred = model(X)
        l = loss(Y, y_pred)

        # gradients = back pass
        l.backward()  # dl/dw

        # update weights
        optimizer.step()

        # Zero grad
        optimizer.zero_grad()

        if epoch % 10 == 0:
            [w, b] = model.parameters()
            print(f"epoch {epoch + 1}: w = {w[0][0].item():.3f} loss = {l:.8f}")
    print(f"prediction after training  f(5) {model(x_test).item():.3f}")


train_model(X, Y, n_iter)
