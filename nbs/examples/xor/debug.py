# %%
""" Demonstrates the easy of integration of a custom layer """
from distutils.log import error
import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F

# Inherit from Function
class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

#######################################################
class MyMSELoss(Function):
    
    @staticmethod
    def forward(ctx, y_pred, y):    
        ctx.save_for_backward(y_pred, y)
        return ( (y - y_pred)**2 ).mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y = ctx.saved_tensors
        grad_input = 2 * (y_pred - y) / y_pred.shape[0]        
        return grad_input, None
#######################################################

class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(2, 2)
        self.linear1 = Linear(2, 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.linear1(x)
        return x

x = torch.Tensor([[0., 0.],
               [0., 1.],
               [1., 0.],
               [1., 1.]])

y = torch.Tensor([0., 1., 1., 0.]).reshape(x.shape[0], 1)

model = BasicModel()
y_pred = model(x)

# mseloss = F.mse_loss
mseloss = MyMSELoss.apply
loss = mseloss(y_pred.view(-1), y)
loss.backward()


# %%
class OutputLayer(Function):

    @staticmethod
    def forward(ctx, input, weights, bias):
        ctx.save_for_backward(input, weights, bias)
        w_times_x= torch.mm(input, weights)
        return torch.add(w_times_x, bias)
    
    @staticmethod
    def backward(ctx, grad_output):  
        # angles = grad_output / (input_size + 1)
        # cinv = torch.conj(weights.T) / torch.square(torch.abs(weights.T))
        # grad_input = angles @ cinv

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        outputs = torch.ones(1, grad_output.size(1))
        grad_output = grad_output / (input.size(1) + 1)
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.mm(weight)
            cinv = (torch.conj(weight) / torch.square(torch.abs(weight))).T
            grad_input = grad_output.mm(cinv)
        if ctx.needs_input_grad[1]:
            x_pinv = torch.linalg.pinv(
                        torch.cat(
                            [torch.ones(1, input.size(0)), input.T[0:]]
                        )
                    ).T
            angle_pinv = x_pinv[1:, :]
            grad_weight = angle_pinv @ torch.div(grad_output, torch.abs(outputs))
            # cinv = (torch.conj(weight) / torch.square(torch.abs(weight))).T
            # grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            angle_pinv = x_pinv[0, :]
            grad_bias = (angle_pinv @ torch.div(grad_output, torch.abs(outputs))).unsqueeze(dim=1)
            # grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class MyLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        # weights = torch.Tensor(size_out, size_in)

        # initialize weights and biases
        weights = torch.randn(
            self.size_in, self.size_out, dtype=torch.cdouble
        ) / math.sqrt(self.size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        
        bias = torch.unsqueeze(
            torch.zeros(size_out, dtype=torch.cdouble, requires_grad=True), 0
        )
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        # w_times_x= torch.mm(x, self.weights)
        # return torch.add(w_times_x, self.bias)  # w times x + b
        return OutputLayer.apply(x, self.weights, self.bias)
        #return x @ self.weights + self.bias


#PyTorch
class Complex_RMSE(nn.Module):
    def __init__(self, categories: int = 2, periodicity: int = 2):
        super(Complex_RMSE, self).__init__()
        self.categories = categories
        self.periodicity = periodicity

    def forward(self, inputs, targets): 
        target_angles = self.class2angle(targets)
        predicted_angles = torch.remainder(inputs.angle() + 2 * np.pi, 2 * np.pi)
        # z = torch.abs(predicted) * torch.exp(1.j * predicted_angles)
        # error calculation

        # errors = torch.exp(1.0j * target_angles) - torch.exp(1.0j * predicted_angles)
        # if self.periodicity > 1:
        #     # select smallest error
        #     idx = torch.argmin(torch.abs(errors), dim=1, keepdim=True)
        #     errors = errors.gather(1, idx)
        #     errors_mean = errors.mean()
        # else:
        #     errors_mean = errors.mean()
        # return errors_mean

        return MyComplexRMSELoss.apply(target_angles, predicted_angles, self.periodicity)

    def class2angle(self, actual: torch.tensor) -> torch.tensor:
        # Returns a new tensor with an extra dimension
        if actual.size().__len__() == 1:
            actual = torch.unsqueeze(actual, 1)

        # Class to several angles due to periodicity using bisector
        return (
            (self.categories * torch.arange(self.periodicity) + actual + 0.5)
            / (self.categories * self.periodicity)
            * 2
            * np.pi
        )

# %%
#######################################################
class MyComplexRMSELoss(Function):

    @staticmethod
    def forward(ctx, target_angles, predicted_angles, periodicity):
        errors = torch.exp(1.0j * target_angles) - torch.exp(1.0j * predicted_angles)
        if periodicity > 1:
            # select smallest error
            idx = torch.argmin(torch.abs(errors), dim=1, keepdim=True)
            errors = errors.gather(1, idx)
            errors_mean = errors.mean()
        else:
            errors_mean = errors.mean()
        ctx.save_for_backward(errors)
        return errors_mean
    
    @staticmethod
    def backward(ctx, grad_output):
        errors = ctx.saved_tensors

        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        grad_input = grad_weight = grad_bias = None
        grad_input = errors[0]
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            pass
            #grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            pass
            #grad_weight = grad_output.t().mm(input)
        if ctx.needs_input_grad[2]:
            pass
            # grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

        return errors[0].mean(), None, None
    
#######################################################

#######################################################
class MyMSELoss(Function):
    
    @staticmethod
    def forward(ctx, y_pred, y, categories, periodicity):    
        y_tmp = y
        if y.size().__len__() == 1:
            y_tmp = torch.unsqueeze(y, 1)

        # Class to several angles due to periodicity using bisector
        target_angles = (
            (categories * torch.arange(periodicity) + y_tmp + 0.5)
            / (categories * periodicity)
            * 2
            * np.pi
        )
        
        predicted_angles = torch.remainder(y_pred.angle() + 2 * np.pi, 2 * np.pi)
        
        # error_angles_list = []
        # for i in range(target_angles.shape[1]):
        #     # error_angles_list.append(target_angles[:, i].unsqueeze(1) - predicted_angles.squeeze())
        #     error_angles_list.append(torch.exp(1.0j * target_angles[:, i].unsqueeze(1)) - torch.exp(1.0j * predicted_angles.squeeze()))

        
        # errors = error_angles_list[0]
        errors = torch.exp(1.0j * target_angles) - torch.exp(1.0j * predicted_angles.unsqueeze(1))

        if periodicity > 1:
            # select smallest error
            idx = torch.argmin(torch.abs(errors), dim=1, keepdim=True)
            errors = errors.gather(1, idx)

        ctx.save_for_backward(y_pred, y, errors)
        return errors.mean()
        # return ( (y - y_pred)**2 ).mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y, errors = ctx.saved_tensors
        #grad_input = 2 * (y_pred - y) / y_pred.shape[0]        
        grad_input = errors.squeeze()
        return grad_input, None, None, None
#######################################################

# %%
class phase_activation(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input / torch.abs(input)
    
    @staticmethod
    def backward(ctx, grad_output):  
        return grad_output, None

class cmplx_phase_activation(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return phase_activation.apply(x)
        #return x @ self.weights + self.bias

# %%
class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = MyLinearLayer(2, 1)
        self.phase_act = cmplx_phase_activation()

    def forward(self, x):
        x = self.linear(x)
        x = self.phase_act(x)
        return x

# %%
torch.manual_seed(0)  #  for repeatable results
model = BasicModel()

# %%
model.linear.weights

# %% [markdown]
# ## XOR Example

# %%
x = torch.Tensor([[0., 0.],
               [0., 1.],
               [1., 0.],
               [1., 1.]])

x = torch.Tensor([[1., 1.],
               [1., -1.],
               [-1., 1.],
               [-1., -1.]])

y = torch.Tensor([0., 1., 1., 0.]).reshape(x.shape[0], 1)

# %%
x

# %%
x = x.type(torch.cdouble)
y_pred = model(x)

# criterion = Complex_RMSE(categories=2, periodicity=2)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

# loss = criterion(y_pred, y)

mseloss = MyMSELoss.apply
loss = mseloss(y_pred.view(-1), y, 2, 2)

loss.backward()

print(model.linear.weights.grad) 
print(model.linear.bias.grad)

optimizer.step()


# %%
print('Forward computation thru model:', model(x))

# %%


# # %%
# # criterion = torch.nn.MSELoss(reduction='sum')
# # criterion = MSEC(categories=2, periodicity=2)
# criterion = Complex_RMSE(categories=2, periodicity=2)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

# # %%
# y_pred = model(x)

# # %%
# y_pred

# # %%
# loss = criterion(y_pred, y)
# loss

# # %%
# loss.backward()

# # %%
# optimizer.step()

# %%

for t in range(2000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    # loss = criterion(y_pred, y)
    loss = mseloss(y_pred.view(-1), y, 2, 2)
    if t % 100 == 99:
        print(t, torch.abs(loss))
        # print(t, torch.square(
        #         np.pi - torch.abs(torch.abs(loss) - np.pi)
        #     ))
        

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# print(f'Result: {model.string()}')

# %%

def angle2class(x: torch.tensor, categories, periodicity) -> torch.tensor:
        tmp = x.angle() + 2 * np.pi
        angle = torch.remainder(tmp, 2 * np.pi)

        # This will be the discrete output (the number of sector)
        o = torch.floor(categories * periodicity * angle / (2 * np.pi))
        return torch.remainder(o, categories)

predictions = model(x)
angle2class(predictions[0], 2, 2)

# %%


# %%
""" Demonstrates the easy of integration of a custom layer """
import math
import torch
import torch.nn as nn
import numpy as np

class MyLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        w_times_x= torch.mm(x, self.weights.t())
        return torch.add(w_times_x, self.bias)  # w times x + b


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 128, 3)
        # self.linear = nn.Linear(256, 2)
        self.linear = MyLinearLayer(256, 2)

    def forward(self, x):
        x = self. conv(x)
        x = x.view(-1, 256)
        return self.linear(x)

torch.manual_seed(0)  #  for repeatable results
basic_model = BasicModel()
inp = np.array([[[[1,2,3,4],  # batch(=1) x channels(=1) x height x width
                  [1,2,3,4],
                  [1,2,3,4]]]])
x = torch.tensor(inp, dtype=torch.float)
print('Forward computation thru model:', basic_model(x))

# %%
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(basic_model.parameters(), lr=1e-6)

# %%
y_pred = basic_model(x)
# Compute and print loss
loss = criterion(y_pred, y)

# %%
y_pred

# %%
loss.backward()


