from sympy import symbols, Matrix, simplify
import torch.nn as nn
from inspect import signature
import torch.nn.functional as F
import torch
import numpy as np  # Import numpy for better handling of array operations
from bases import *

def get_model_equation(model, arg_max=True):
    def argmax_matrix(M):
        # Ensure the input M is a tensor
        if not isinstance(M, torch.Tensor):
            M = torch.tensor(M, dtype=torch.float32)
        argmaxes = torch.argmax(M, dim=1).unsqueeze(-1)
        matrix = torch.zeros_like(M)
        for i, argmax in enumerate(argmaxes):
            matrix[i, argmax] = 1
        return matrix

    inputs = symbols(['x_' + str(i) for i in range(model.number_of_variables)])
    constants = symbols([LATEX_CONSTANTS[constant][1:][:-1] for constant in model.constants])
    outputs = symbols(['y_' + str(i) for i in range(model.number_of_outputs)])

    bases = [SYMPY_BASES[base] for base in model.bases]

    layers = []
    for module in model.children():
        layers += [l for l in module.children()] if isinstance(module, nn.ModuleList) else [module]

    source = layers[0]
    source_w = source.weight.detach().cpu().numpy()  # Convert to numpy array

    number_of_inputs = source_w.shape[1]
    input_variables = Matrix(inputs + constants)

    # Ensure the softmax operation is applied to a torch tensor
    source_w_tensor = torch.tensor(source_w, dtype=torch.float32)
    source_w_tensor = F.softmax((1.0 / model.temperature) * source_w_tensor, dim=1)
    source_w_tensor = argmax_matrix(source_w_tensor)
    
    # Convert the tensor back to numpy for sympy
    source_w = Matrix(source_w_tensor.cpu().numpy().astype(float))  # Ensure it's a numerical matrix

    args = source_w * input_variables
    past_imgs = inputs + constants

    for layer in layers[1:]:
        args_idx, img = 0, []
        for f in bases:
            arity = get_arity(f)
            arg = args[args_idx: args_idx + arity]
            img.append(f(*arg))
            args_idx = args_idx + arity

        if model.skip_connections:
            img = Matrix(img + past_imgs)
        else:
            img = Matrix(img)

        past_imgs = img[:]
        W = layer.weight.detach().cpu().numpy()  # Convert to numpy array
        W_tensor = torch.tensor(W, dtype=torch.float32)
        W_tensor = F.softmax((1.0 / model.temperature) * W_tensor, dim=1)
        W_tensor = argmax_matrix(W_tensor)

        # Convert the tensor back to numpy for sympy
        W = Matrix(W_tensor.cpu().numpy().astype(float))  # Ensure it's a numerical matrix
        args = W * img

    args = [simplify(arg) for arg in args]
    return args

def get_arity(f):
    return len(signature(f).parameters)
