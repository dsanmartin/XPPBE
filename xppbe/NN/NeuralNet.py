import numpy as np
import torch
import torch.nn as nn


class PINN_2Dom_NeuralNet(nn.Module):

    def __init__(self, hyperparameters, bc_param=None, **kwargs):
        super().__init__()
        param_1, param_2 = hyperparameters
        if bc_param is None:
            self.NNs = nn.ModuleList([NeuralNet(**param_1), NeuralNet(**param_2)])
        else:
            self.NNs = nn.ModuleList([NeuralNet(**param_1), NeuralNet_constrained(param_2, bc_param)])
        

    def forward(self, X, flag):
        if flag == 'molecule':
            output = self.NNs[0](X)
            outputs = torch.cat([output, torch.zeros_like(output)], dim=1)
        elif flag == 'solvent':
            output = self.NNs[1](X)
            outputs = torch.cat([torch.zeros_like(output), output], dim=1)
        elif flag =='interface':
            outputs = torch.cat([self.NNs[0](X), self.NNs[1](X)], dim=1)
        return outputs
    
    def build_Net(self):
        pass  # Not needed in PyTorch


class PINN_1Dom_NeuralNet(nn.Module):

    def __init__(self, hyperparameters, bc_param=None, **kwargs):
        super().__init__()
        param_1, param_2 = hyperparameters
        self.NN = NeuralNet(**param_1)
        self.NNs = [self.NN, self.NN]

    def forward(self, X, flag):
        output = self.NN(X)
        outputs = torch.cat([output, output], dim=1)
        return outputs
    
    def build_Net(self):
        pass  # Not needed in PyTorch


class NeuralNet_constrained(nn.Module):
    
    def __init__(self, hyperparameters, bc_param, **kwargs):
        super().__init__() 
        self.fun = bc_param['fun']
        self.R_sphere = float(bc_param['R'])
        self.NN = NeuralNet(**hyperparameters)
        self.input_shape_N = self.NN.input_shape_N

    def forward(self, X):
        output = self.NN(X) * (self.R_sphere - torch.norm(X, dim=1, keepdim=True)) / self.R_sphere + self.fun(X)
        return output
    
    def build_Net(self):
        pass  # Not needed in PyTorch
        

class NeuralNet(nn.Module):

    DTYPE = 'float32'

    def __init__(self, 
                 input_shape=(None, 3),
                 output_dim=1,
                 num_hidden_layers=2,
                 num_neurons_per_layer=20,
                 num_hidden_blocks=2,
                 activation='tanh',
                 adaptive_activation=False,
                 kernel_initializer='glorot_normal',
                 architecture_Net='FCNN',
                 fourier_features=False, 
                 num_fourier_features=128, 
                 fourier_sigma=1,
                 weight_factorization=False,
                 scale_input=True,
                 scale_output=False,
                 scale_input_s=[[-1.,-1.,-1.],[1.,1.,1.]],
                 scale_output_s=[-1.,1.],
                 **kwargs):
        super().__init__()
        
        # Store input_dim from input_shape
        self.input_dim = input_shape[-1] if isinstance(input_shape, (list, tuple)) else input_shape

        self.input_shape_N = input_shape
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        self.num_hidden_blocks = num_hidden_blocks
        
        self.activation = activation
        self.adaptive_activation = adaptive_activation

        self.kernel_initializer = kernel_initializer
        self.architecture_Net = architecture_Net
        self.use_fourier_features = fourier_features
        self.num_fourier_features = num_fourier_features
        self.fourier_sigma = fourier_sigma
        self.weight_factorization = weight_factorization
 
        self.scale_input = scale_input
        dtype = torch.float32 if self.DTYPE == 'float32' else torch.float64
        self.register_buffer('input_lb', torch.tensor(scale_input_s[0], dtype=dtype))
        self.register_buffer('input_ub', torch.tensor(scale_input_s[1], dtype=dtype))

        self.scale_output = scale_output
        self.register_buffer('output_lb', torch.tensor(scale_output_s[0], dtype=dtype))
        self.register_buffer('output_ub', torch.tensor(scale_output_s[1], dtype=dtype))


        self.weight_factorization_flag = weight_factorization
        self.Dense_Layer = CustomDenseLayer if weight_factorization else nn.Linear

        # Fourier feature layer
        if self.use_fourier_features:
            self.fourier_layer = nn.Linear(self.input_dim, num_fourier_features, bias=False)
            nn.init.normal_(self.fourier_layer.weight, std=self.fourier_sigma)
            self.fourier_layer.weight.requires_grad = False
        

        if self.architecture_Net in ('FCNN','MLP'):
            self.create_FCNN()
           
        elif self.architecture_Net == 'ModMLP':
            self.create_ModMLP()

        elif self.architecture_Net == 'ResNet':
            self.create_ResNet()

        # Output layer
        if self.weight_factorization_flag:
            self.out = self.Dense_Layer(self.num_neurons_per_layer, output_dim, self.kernel_initializer)
        else:
            self.out = nn.Linear(self.num_neurons_per_layer, output_dim)
  

    def create_FCNN(self):
        self.hidden_layers = nn.ModuleList()
        in_dim = self.num_fourier_features * 2 if self.use_fourier_features else self.input_dim
        for i in range(self.num_hidden_layers):
            if self.weight_factorization_flag:
                layer = self.Dense_Layer(in_dim if i == 0 else self.num_neurons_per_layer,
                                        self.num_neurons_per_layer,
                                        self.kernel_initializer)
            else:
                layer = self.Dense_Layer(in_dim if i == 0 else self.num_neurons_per_layer,
                                        self.num_neurons_per_layer)
            self.hidden_layers.append(layer)
        self.activation_fn = CustomActivation(self.num_neurons_per_layer, self.activation, self.adaptive_activation)
        self.call_architecture = self.call_FCNN

    def create_ModMLP(self):
        self.create_FCNN()
        in_dim = self.num_fourier_features * 2 if self.use_fourier_features else self.input_dim
        if self.weight_factorization_flag:
            self.U = self.Dense_Layer(in_dim, self.num_neurons_per_layer, self.kernel_initializer)
        else:
            self.U = self.Dense_Layer(in_dim, self.num_neurons_per_layer)
        if self.weight_factorization_flag:
            self.V = self.Dense_Layer(in_dim, self.num_neurons_per_layer, self.kernel_initializer)
        else:
            self.V = self.Dense_Layer(in_dim, self.num_neurons_per_layer)
        self.call_architecture = self.call_ModMLP


    def create_ResNet(self):
        in_dim = self.num_fourier_features * 2 if self.use_fourier_features else self.input_dim
        if self.weight_factorization_flag:
            self.first = self.Dense_Layer(in_dim, self.num_neurons_per_layer, self.kernel_initializer)
        else:
            self.first = self.Dense_Layer(in_dim, self.num_neurons_per_layer)
        self.first_activation = CustomActivation(self.num_neurons_per_layer, self.activation, self.adaptive_activation)
        
        self.hidden_blocks = nn.ModuleList()
        self.hidden_blocks_activations = nn.ModuleList()
        for i in range(self.num_hidden_blocks):
            block = nn.ModuleList()
            if self.weight_factorization_flag:
                block.append(self.Dense_Layer(self.num_neurons_per_layer, self.num_neurons_per_layer, self.kernel_initializer))
            else:
                block.append(self.Dense_Layer(self.num_neurons_per_layer, self.num_neurons_per_layer))
            block.append(CustomActivation(self.num_neurons_per_layer, self.activation, self.adaptive_activation))
            if self.weight_factorization_flag:
                block.append(self.Dense_Layer(self.num_neurons_per_layer, self.num_neurons_per_layer, self.kernel_initializer))
            else:
                block.append(self.Dense_Layer(self.num_neurons_per_layer, self.num_neurons_per_layer))
            self.hidden_blocks.append(block)
            activation_layer = CustomActivation(self.num_neurons_per_layer, self.activation, self.adaptive_activation)
            self.hidden_blocks_activations.append(activation_layer)
        
        if self.weight_factorization_flag:
            self.last = self.Dense_Layer(self.num_neurons_per_layer, self.num_neurons_per_layer, self.kernel_initializer)
        else:
            self.last = self.Dense_Layer(self.num_neurons_per_layer, self.num_neurons_per_layer)
        self.last_activation = CustomActivation(self.num_neurons_per_layer, self.activation, self.adaptive_activation)
        self.call_architecture = self.call_ResNet


    def build_Net(self):
        pass  # Not needed in PyTorch

    def forward(self, X):
        if self.scale_input:
            X = 2.0 * (X - self.input_lb) / (self.input_ub - self.input_lb) - 1.0
        if self.use_fourier_features:
            X = self.fourier_layer(X)
            X = torch.cat([torch.sin(2.0 * np.pi * X), torch.cos(2.0 * np.pi * X)], dim=-1)
        X = self.call_architecture(X)
        X = self.out(X)
        if self.scale_output:
            X = (X + 1.0) / 2.0 * (self.output_ub - self.output_lb) + self.output_lb
        return X

    def call_FCNN(self, X):
        for layer in self.hidden_layers:
            X = self.activation_fn(layer(X))
        return X

    def call_ModMLP(self, X):
        U = self.activation_fn(self.U(X))
        V = self.activation_fn(self.V(X))
        for layer in self.hidden_layers:
            out = self.activation_fn(layer(X))
            X = out * U + (1 - out) * V
        return X

    def call_ResNet(self, X): 
        X = self.first_activation(self.first(X))
        for block, activation in zip(self.hidden_blocks, self.hidden_blocks_activations):
            residual = X
            X = block[1](block[0](X))
            X = block[2](X)
            X = activation(X + residual)
        return self.last_activation(self.last(X))
    

class CustomActivation(nn.Module):

    def __init__(self, units=1, activation='tanh', adaptive_activation=False, **kwargs):
        super(CustomActivation, self).__init__()
        self.units = units
        self.activation_name = activation
        self.adaptive_activation = adaptive_activation
        
        self.a = nn.Parameter(torch.ones(units), requires_grad=adaptive_activation)
        
        # Map activation names to PyTorch functions
        activation_map = {
            'tanh': torch.tanh,
            'relu': torch.relu,
            'sigmoid': torch.sigmoid,
            'elu': torch.nn.functional.elu,
            'softplus': torch.nn.functional.softplus,
            'swish': lambda x: x * torch.sigmoid(x),
            'gelu': torch.nn.functional.gelu
        }
        self.activation_func = activation_map.get(activation, torch.tanh)

    def forward(self, inputs):
        return self.activation_func(inputs * self.a.unsqueeze(0))


class CustomDenseLayer(nn.Module):

    def __init__(self, input_dim, units, kernel_initializer='glorot_normal', **kwargs):
        super(CustomDenseLayer, self).__init__()
        self.input_dim = input_dim
        self.units = units
        
        # Initialize weights
        if kernel_initializer == 'glorot_normal':
            W = torch.empty(input_dim, units)
            nn.init.xavier_normal_(W)
        elif kernel_initializer == 'glorot_uniform':
            W = torch.empty(input_dim, units)
            nn.init.xavier_uniform_(W)
        else:
            W = torch.randn(input_dim, units) * 0.05
        
        S, V = self.weight_factorization(W)
        self.S = nn.Parameter(S, requires_grad=True)
        self.V = nn.Parameter(V, requires_grad=True)
        self.b = nn.Parameter(torch.zeros(units), requires_grad=True)

    def weight_factorization(self, W, mean=1.0, stddev=0.1):
        S = mean + torch.randn(W.shape[-1]) * stddev
        S = torch.exp(S)
        V = W / S
        return S, V

    def forward(self, inputs):
        SV = self.S * self.V
        outputs = torch.matmul(inputs, SV) + self.b
        return outputs

