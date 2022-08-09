# %% [markdown]
# # Standard PyTorch MSE loss function

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# Let's generate some fake data
torch.manual_seed(42)
resid = torch.rand(100)    
inputs = torch.tensor([ [ xx ] for xx in range(100)] , dtype=torch.float32)
labels = torch.tensor([ (2 + 0.5*yy + resid[yy]) for yy in range(100)], dtype=torch.float32)

# Now we define a linear regression model
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.bn = torch.nn.BatchNorm1d(num_features=1)
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, inx):
        x = self.bn(inx) # Adding BN to standardize input helps us use a higher learning rate
        x = self.linear(x)
        return x


# %%
from torch.autograd import Function

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
    
model = linearRegression(1, 1) 
epochs = 25    
mseloss = MyMSELoss.apply
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# %%
model.train()
outputs = model(inputs)
loss = mseloss(outputs.view(-1), labels)

# %%
loss

# %%
loss.backward()

# %%
for epoch in range(epochs):
    model.train()
    outputs = model(inputs)
    loss = mseloss(outputs.view(-1), labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    print(f'epoch {epoch}, loss {loss}')


