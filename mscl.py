import torch

#================================#
#       TENSOR INITIALIZATION    #
#================================#

# im on mac m2
mps_device = torch.device("mps")

my_tensor = torch.tensor([[1, 1, 3], 
                         [4, 5, 6]], dtype=torch.float32, device="mps") # requires_grad=True, device="mps" are other params

# some attributes of a torch.tensor
print(my_tensor)
print(my_tensor.device)
print(my_tensor.dtype)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common initialization methods
x = torch.empty(size = (3, 3)) # values inside the tensor are the one that are currently in memory
x = torch.zeros((3, 3)) # for initalizing a tensor full of 0s
x = torch.rand((3, 3)) # tensor with random values
x = torch.ones((3, 3)) # tensor full of 1's
x = torch.eye(5, 5) # identity matrix (1 in diagonale)
x = torch.arange(start=0, end=5, step=1) 
x = torch.linspace(start=0.1, end=1, steps=10) # go from 0.1 to 1 and have 10 values (constant steps)
# tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000, 1.0000])
x = torch.empty((1,5)).normal_(mean=0, std=1) # create a tensor with vals that respect these stats
# tensor([[ 0.4158,  0.6784, -0.6709,  1.8475,  0.3635]])
x = torch.diag(torch.ones(3)) # 3x3 diagonal matrix -> same as identity matrix
print(x)


#================================#
#       CASTING TENSOR TYPES     #
#================================#

tensor = torch.arange(4) # start = 0 by def., step=1 by def.
print(tensor.bool()) # works on CUDA and CPU, 0 = False, rest = True
print(tensor.short()) # to short
print(tensor.half()) # to float16
print(tensor.float()) # to float32 (important)
print(tensor.double()) # to double/float64

# array to tensor conversion and vice versa
import numpy as np

np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array) # convert np array to tensor
np_array_back = tensor.numpy() # convert back to np array


#========================================#
#       TENSOR MATH & COMPARISON OPS     #
#========================================#

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1) # Method 1
z2 = torch.add(x, y) # Method 2
z = x + y # Method 3

# Subtraction
z = x - y

# Division
z = torch.true_divide(x, y) # divide each item in x by corresponding item in y
z = torch.true_divide(x, 2) # divide all items in x by 2
print(z)

# inplace operations ("{ops}_" indicate inplace)
t = torch.zeros(3)
t.add_(x) # inplace ops, no need for other var so more efficient
t += x # inplace too but t = t + x is not inplace

# Exponentiation
z = x.pow(2)
z = x ** 2
print(z)

# Simple comparison
z = x > 0
z = x < 0
print(z) # element wise comparison

# Matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # 2x3
x3 = x1.mm(x2) # works too


# Matrix exponentiation (raise entire matrix)
matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_power(3))

# Element wise multiplication
z = x * y 
print(z)

# Dot product
z = torch.dot(x, y)
print(z)

# Batch Matrix Multiplication (more efficient)
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p)) # dim 
out_bmm = torch.bmm(tensor1, tensor2) # (batch, n, p)
print(out_bmm)

# Broadcasting
x1 = torch.ones((5, 5))
x2 = torch.ones((1, 5))

z = x1 - x2 # How is that possible?? Expand x2's dim to match x1 automatically
print(z)

# Other useful tensor ops
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0) # gives val and pos of max, 1st occurence
values, indices = torch.min(x, dim=0) # gives val and pos of min, 1st occurence
abs_x = torch.abs(x)  # element wise abs value
z = torch.argmax(x, dim=0) # gives min value
z = torch.argmin(x, dim=0) # gives max value
mean_x = torch.mean(x.float(), dim=0) # returns scalar mean
z = torch.eq(x, y) # check element wise if elements are equal 
sorted_y, indices = torch.sort(y, dim=0, descending=False) # sorted tensor + new indices order
z = torch.clamp(x, min=0, max=10) # if value below 0, set to 0, if value > 10, set to 10

x = torch.tensor([1, 0, 1, 1], dtype=torch.bool) # bool tensor
z = torch.any(x) # True
z = torch.all(x) # False, we have one 0

#========================================#
#            TENSOR INDEXING             #
#========================================#

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape) # x[0,:], all features of first example/batch
print(x[:, 0].shape) # x[:, 0] first feature of all our examples, ":" means all
# out:
# torch.Size([25])
# torch.Size([10])
print(x[2, 0:10]) # get 10 first features of the 2nd example

# set the val of the tensor at an index
x[0, 0] = 100

# Fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices]) # pick 3 elements from x: 3rd pos, 6th pos and 9th pos
# out: tensor([2, 5, 8])

x = torch.rand(3, 5)
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x)
print(x[rows, cols]) # looks for x[1, 4] and x[0, 0]
# out: torch.Size([2])

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x >8)]) # elems of x at pos 0, 1 and 9
print(x[x.remainder(2) == 0]) # keep only even numbers
# out: tensor([0, 2, 4, 6, 8])

# Useful ops
print(torch.where(x > 5, x, x*2)) # check elemt wise if > 5, if yes, let it as is, else multiply it by 2
# out: tensor([ 0,  2,  4,  6,  8, 10,  6,  7,  8,  9])

print(torch.tensor([0,0,1,1,2,4]).unique()) # remove duplicates
print(x.ndimension()) # out: 1 since only one dim
print(x.numel()) # len num of elements in x


#========================================#
#            TENSOR RESHAPING            #
#========================================#

x = torch.arange(9)

x_3x3 = x.view(3, 3) # Method 1: for contiguous tensor
x_3x3 = x.reshape(3, 3) # Method 2: safer, handle both cases
print(x_3x3.shape)

y = x_3x3.t() # do the transpose, not contiguous here

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0).shape)
# out: torch.Size([4, 5])
z = x1.view(-1) # flatten tensor
print(z)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1) # just flatten the 2, 5 part
print(z.shape)
# out: torch.Size([64, 10])

# swap axis
z = x.permute(0, 2, 1) # swap dim 2 (5) with dim 1 (2 up here)
print(z.shape)

x = torch.arange(10) # dim 10 rn, we want dim 1x10
print(x.unsqueeze(0).shape) # out: torch.Size([1, 10])

x = torch.arange(10).unsqueeze(0).unsqueeze(1) # 1x1x10

z = x.squeeze(1)
print(z.shape)
# out: torch.Size([1, 10])
