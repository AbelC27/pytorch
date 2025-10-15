import torch

# we use ones for testing and checking data it is correct
x = torch.ones((2, 3))
print(x)

# we use zeros to initialize layers on my model
x = torch.zeros((2, 3))
print(x)

# we use to create random data between 0 and 1 good for non-constant data
x = torch.rand((2, 2))
print(x)

# mean - calculate the average value of all elements in a tensor
x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
print(x.mean())

# reshape - rearrange data for operations
x = torch.tensor([1, 2, 3, 4, 5, 6])
print(x)
y = x.reshape(2, 3)
print(y)

#dtype - check the type of data, long float32, int, bool, etc, good for shifting from int to float when needed!
x = torch.tensor([1, 2, 3, 4, 5, 6])
print(x.dtype)
x = x.type(torch.float32)
print(x)
print(x.dtype)
# we can also create tensors directly with a specific type
x = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
print(x)
print(x.dtype)

x = torch.ones((3, 3))
print(x)
y = torch.rand(3, 3)
print(y)
print(y.dtype, x.dtype)

x = torch.tensor([1, 2, 3, 4, 5, 6])
x = x.type(torch.float32)
x = x.reshape(2, 3)
print(x.mean())

x = torch.tensor([1, 2, 3])
print(x.dtype)
x = x.type(torch.float32)
print(x.dtype)

x = torch.rand(2, 4)
y = x.reshape(4, 2)

print(x, y)
print(x.shape, y.shape)
#checking number elements
print(x.numel(),y.numel())
#check to see if the values are identical
print(torch.equal(x.reshape(-1),y.reshape(-1)))

