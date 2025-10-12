import torch

# we use ones for testing and checking data it is correct
x = torch.ones((2,3))
print(x)

# we use zeros to initialize layers on my model
x= torch.zeros((2,3))
print(x)

# we use to create random data between 0 and 1
x = torch.rand((2,2))
print(x)