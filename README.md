MLMVN
================

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

## Install

Clone this repository

    git clone https://github.com/antonpf/mlmvn.git

and install the required packages

    cd mlmvn
    conda env create -f environment.yml
    conda activate mlmvn

Next, the mlmvn package can be installed with pip

    pip install .

## How to use

As a small example, the XOR problem is described here. The XOR problem
is an example of how a single real-valued neuron cannot learn a simple
but non-linear relationship. At least, this holds if we do not extend
the dimensionality of the feature space.

### Setup

``` python
import torch
import torch.nn as nn
import numpy as np
from mlmvn.layers import FirstLayer, OutputLayer, cmplx_phase_activation
from mlmvn.loss import ComplexMSELoss
from mlmvn.optim import ECL
```

### Loading Data

The dataset contains four input-output mappings with binary classes. The
two-dimensional input $x$ is mapped to a class label $y$. The following
table shows the truth table with associated labels for the XOR gate.

$$
\begin{aligned}
    \begin{array}{cc|c|cc}
        x_1 & x_2 & y & z & arg(z) \\
        \hline
        1 &  1  & 0 &  1+j &  45° \\
        1 & -1  & 1 &  1-j & 315° \\
        -1 &  1 & 1 & -1+j & 135° \\
        -1 & -1 & 0 & -1-j & 225° \\
    \end{array}
\end{aligned}
$$

``` python
# Create data
x = torch.Tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
x = x.type(torch.cdouble)
y = torch.Tensor([0.0, 1.0, 1.0, 0.0]).reshape(x.shape[0], 1)
```

### Creating Models

``` python
class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = FirstLayer(2, 2)
        self.phase_act = cmplx_phase_activation()
        self.linear1 = OutputLayer(2, 1)
        self.phase_act = cmplx_phase_activation()

    def forward(self, x):
        x = self.linear(x)
        x = self.linear1(x)
        x = self.phase_act(x)
        return x
```

``` python
model = BasicModel()
criterion = ComplexMSELoss.apply
optimizer = ECL(model.parameters(), lr=1)
categories = 2
periodicity = 2
```

## Training

``` python
for t in range(5):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)
    loss = criterion(y_pred, y, categories, periodicity)
    print(t, torch.abs(loss))

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(inputs=x, layers=list(model.children()))
```

    0 tensor(0.2833, dtype=torch.float64, grad_fn=<AbsBackward0>)
    1 tensor(1.3938, dtype=torch.float64, grad_fn=<AbsBackward0>)
    2 tensor(0.3198, dtype=torch.float64, grad_fn=<AbsBackward0>)
    3 tensor(0.0371, dtype=torch.float64, grad_fn=<AbsBackward0>)
    4 tensor(0.0036, dtype=torch.float64, grad_fn=<AbsBackward0>)

### Evaluation

``` python
predictions = model(x)


def angle2class(x: torch.tensor, categories, periodicity) -> torch.tensor:
    tmp = x.angle() + 2 * np.pi
    angle = torch.remainder(tmp, 2 * np.pi)

    # This will be the discrete output (the number of sector)
    o = torch.floor(categories * periodicity * angle / (2 * np.pi))
    return torch.remainder(o, categories)


angle2class(predictions, 2, 2)
```

    tensor([[0.],
            [1.],
            [1.],
            [0.]], dtype=torch.float64, grad_fn=<RemainderBackward0>)
