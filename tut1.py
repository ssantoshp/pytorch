import torch

#================================#
#       TENSOR INITIALIZATION    #
#================================#

# im on mac m2
mps_device = torch.device("mps")

my_tensor = torch.tensor([[1, 2, 3], 
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




