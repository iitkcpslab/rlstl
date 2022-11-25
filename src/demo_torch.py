# importing torch
import torch

# creating a tensor
t1=torch.tensor(1.0, requires_grad = True)
t2=torch.tensor(2.0, requires_grad = True)

# creating a variable and gradient
z=100 * t1 * t2
t1=torch.tensor(0.0, requires_grad = True)
z=torch.abs(t1)
z.backward()

# printing gradient
print("dz/dt1 : ", t1.grad.data)
print("dz/dt1 : ", t1.grad)
#print("dz/dt2 : ", t2.grad.data)

x = torch.tensor(1.0, requires_grad=True)
x.norm().backward()
print(x.grad)

h=torch.tensor([1, float('nan'), 2])
h=torch.tensor([1, float(1), 2])
h=torch.tensor([float('nan'), float('nan'), float('nan')])
print(torch.isnan(h))
print(torch.all(~torch.isnan(h))) # this is false when either of the tensor element is nan


a = torch.randn((), requires_grad=True)
b = torch.tensor(False)
c = torch.ones(())

print(torch.where(b, a/0, c))
print(torch.autograd.grad(torch.where(b, a/0, c), a))

# This behavior underlies the fix to clamp, which uses where in its derivative
x = torch.tensor([-10., -5, 0, 5, 10, 50, 60, 70, 80, 90, 100], requires_grad=True)
#y = torch.where(x > 0, x, -1*x-1)
y = torch.abs(x)
print("y:", y)
print(y.grad_fn)
y.sum().backward()
print("x.grad:", x.grad)


from torch.autograd import gradcheck

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
from torch.autograd import Variable

dtype = torch.FloatTensor

N=3
D_in=1
x1 = Variable(torch.randn(N, D_in).type(dtype), requires_grad=True)

#x = torch.tensor([0.0, -5.0, 5.0], requires_grad=True)
x = torch.tensor([10.0, 15.0, 5.0], requires_grad=True)
#y = torch.where(x > 0, x, -1*x-1)
y = torch.abs(x)
#test = gradcheck(torch.abs(x1), x, eps=1e-6, atol=1e-4)
test = gradcheck(torch.min, x, eps=1e-6, atol=1e-4)
print(test)
'''
# Let's say we want to preprocess some saved weights and use
# the result as new weights.
saved_weights = [0.1, 0.2, 0.3, 0.25]
weights = torch.tensor(saved_weights)
#weights = np.square(loaded_weights)  # some function
print(weights)

# Now, start to record operations done to weights
weights.requires_grad_()
out = weights.pow(2).sum()
out.backward()
print(weights.grad)
'''
