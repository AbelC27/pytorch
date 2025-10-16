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

#Create a tensor with gradient tracking enabled
x=torch.tensor(2.0, requires_grad=True)

#define a function of x
y=x**2+3*x+1

#we compute the gradient
y.backward()

#now we print the gradient
print(x.grad)

a=torch.tensor(3.0, requires_grad=True)
b=a*2
#b will always have grad because a has
print(b.requires_grad)

#if we do not want to have grad we do this for saving memory and time
with torch.no_grad():
    pass

x=torch.tensor(3.0, requires_grad=True)
y=x**2
z=2*y+5

#we compute z now
z.backward()
print("x=",x)
print("z=",z)
print("Gradient (dz/dx)",x.grad)

#shor exercise to learn
x=torch.tensor(0.0, requires_grad=True)
learning_rate=0.1

for step in range(30):
    y=(x-5)**2
    y.backward() #compute loss
    with torch.no_grad(): #update x without tracking
        x-=learning_rate * x.grad # move opposite to gradient
    x.grad.zero_() #reset gradient for next step/iteration
    print(f"step {step+1}: x = {x.item():.4f}, y= {y.item():.4f}")

#exercises
#1
learning_rate=0.001
x= torch.tensor(4.0, requires_grad=True)
for step in range(30):
    y=3*x**3+x**2+2*x+1
    y.backward()
    with torch.no_grad():
        x-=learning_rate * x.grad
    x.grad.zero_()
    print(f"step {step+1}: x= {x.item():.4f}, y= {y.item():.4f}")
print("Ex 1 is done")
#2
x= torch.tensor(2.0, requires_grad=True)
for step in range(30):
    y=x**3
    z=3*y+2
    z.backward()
    with torch.no_grad():
        x-=learning_rate * x.grad
        print(x.grad)
    x.grad.zero_()
    print(f"step {step+1}: x= {x.item():.4f}, y= {y.item():.4f}")
print("Ex 2 is done")
x=torch.tensor(0.0,requires_grad=True)
learning_rate=0.2
for step in range(30):
    y=(x-7)**2
    y.backward()
    with torch.no_grad():
        x-=learning_rate * x.grad
    x.grad.zero_()
    if (step+1)%5==0:
        print(f"step {step+1}: x= {x.item():.4f}, y= {y.item():.4f}")
print("Ex 3 is done")

import torch
import matplotlib.pyplot as plt
import numpy as np


# make figures display inline if you're using notebooks
# %matplotlib inline

def gradient_descent_visual(func, start_x, learning_rate, steps, title):
    """
    func: a Python function that accepts a torch.Tensor and returns a scalar tensor
    """
    # container for plotting
    xs_hist, ys_hist = [], []

    x = torch.tensor(start_x, dtype=torch.float32, requires_grad=True)

    for i in range(steps):
        y = func(x)
        y.backward()  # compute gradient

        # record for history
        xs_hist.append(x.item())
        ys_hist.append(y.item())

        with torch.no_grad():
            x -= learning_rate * x.grad
        x.grad.zero_()

    # plot function curve
    xs = np.linspace(min(xs_hist) - 1, max(xs_hist) + 1, 200)
    ys = [func(torch.tensor(v)).item() for v in xs]

    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, label='Function')
    plt.scatter(xs_hist, ys_hist, color='red', label='Steps')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Final x =", xs_hist[-1], "   Final y =", ys_hist[-1])


# Example 1: a simple quadratic (nice and stable)
gradient_descent_visual(lambda x: (x - 5) ** 2, start_x=0.0,
                        learning_rate=0.1, steps=25,
                        title="Descent on (x - 5)^2")

# Example 2: same idea but a cubic with small LR
gradient_descent_visual(lambda x: 3 * x ** 3 + x ** 2 + 2 * x + 1, start_x=4.0,
                        learning_rate=0.001, steps=40,
                        title="Descent on 3x^3 + x^2 + 2x + 1 (stable LR)")

