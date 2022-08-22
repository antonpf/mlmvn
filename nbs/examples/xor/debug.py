# %%
#|hide
from nbdev.showdoc import *

# %% [markdown]
# # XOR
# 
# > An example of how to solve the XOR problem with MLMVN.

# %% [markdown]
# The XOR problem is an example of how a single real-valued neuron cannot learn a simple but non-linear relationship. At least, this holds if we do not extend the dimensionality of the feature space.

# %%
#|hide
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from mlmvn.layers import OutputLayer, cmplx_phase_activation
from mlmvn.loss import ComplexMSELoss

torch.manual_seed(0)  #  for repeatable results

# %%
config = dict(
    epochs=20,
    classes=2,
    kernels=[2],
    batch_size=4,
    learning_rate=1,
    dataset="XOR",
    architecture="MLMVN")

# %% [markdown]
# The dataset contains four input-output mappings with binary classes. The two-dimensional input $x$ is mapped to a class label $y$. The following table shows the truth table with associated labels for the XOR gate.
# 
# $$
# \begin{aligned}
#     \begin{array}{cc|c|cc}
#         x_1 & x_2 & y & z & arg(z) \\
#         \hline
# 		1 &  1	& 0	&  1+j &  45° \\
# 		1 & -1	& 1	&  1-j & 315° \\
# 		-1 &  1	& 1	& -1+j & 135° \\
# 		-1 & -1	& 0	& -1-j & 225° \\
#     \end{array}
# \end{aligned}
# $$
# 
# 
# If we consider $x_1$ as $Re(z)$ and $x_2$ as $Im(z)$, the problem can also be expressed graphically into the complex domain. 
# 
# <center>
#     <img src="fig/xor_complex_domain.png" width=450 />
# </center>

# %%
# create data
x = torch.Tensor([[1., 1.],
               [1., -1.],
               [-1., 1.],
               [-1., -1.]])

x = x.type(torch.cdouble)

y = torch.Tensor([0., 1., 1., 0.]).reshape(x.shape[0], 1)


# %%
class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = OutputLayer(2, 20)
        self.phase_act = cmplx_phase_activation()
        self.linear1 = OutputLayer(20, 1)
        self.phase_act = cmplx_phase_activation()

    def forward(self, x):
        x = self.linear(x)
        x = self.linear1(x)
        x = self.phase_act(x)
        return x

# %%
model = BasicModel()
criterion = ComplexMSELoss.apply
optimizer = torch.optim.SGD(model.parameters(), lr=1)
categories =  2
periodicity = 2

# %%
for t in range(200):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    loss = criterion(y_pred.view(-1), y, categories, periodicity)
    # wandb.log({"loss": torch.abs(loss)})
    
    if t % 10 == 9: print(t, torch.abs(loss))

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # for idx, param in enumerate(model.parameters()):
    #     wandb.log({"weights_layer"+str(idx)+"_real": param.real})
    #     wandb.log({"weights_layer"+str(idx)+"_imag": param.imag})

# %%
for idx, param in enumerate(model.parameters()):
    param.real

# %%
predictions = model(x)

# %%
def angle2class(x: torch.tensor, categories, periodicity) -> torch.tensor:
    tmp = x.angle() + 2 * np.pi
    angle = torch.remainder(tmp, 2 * np.pi)

    # This will be the discrete output (the number of sector)
    o = torch.floor(categories * periodicity * angle / (2 * np.pi))
    return torch.remainder(o, categories)

angle2class(predictions, 2, 2)


