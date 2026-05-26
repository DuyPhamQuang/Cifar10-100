import torch
import torch.nn as nn
from torch.autograd import Function


class ModifiedReluFunc(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.set_materialize_grads(False)
        ctx.x = x
        ctx.alpha = alpha
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None
        return grad_output * ctx.x.mul(ctx.alpha).sigmoid(), None


class ModifiedRelu(nn.Module):
    def __init__(self, alpha):
        super(ModifiedRelu, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return ModifiedReluFunc.apply(x, self.alpha)


class MLP(nn.Module):
    def __init__(self, input_dim, model_hidden_list, output_dim, reconstruction_model_relu_alpha, use_bias=False):
        super().__init__()
        self.activation = ModifiedRelu(reconstruction_model_relu_alpha)
        self.layers = nn.ModuleList([nn.Linear(input_dim, model_hidden_list[0])])  # input layer

        for i in range(1, len(model_hidden_list)):
            self.layers.append(nn.Linear(model_hidden_list[i-1], model_hidden_list[i], bias=use_bias))   # hidden layers   

        self.layers.append(nn.Linear(model_hidden_list[-1], output_dim, bias=False))  # output layer

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # flatten the input
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

