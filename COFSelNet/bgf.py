import torch
import math

def f1(x, p=None):  
    # Rastrigin’s function
    # [1, 10, 10, 1, 1] are the default parameter values if 'p' is not provided

    if p is None:
        p = torch.tensor([1, 10, 10, 1, 1], dtype=torch.float32)
    # x[0],x[1]=x[0]+0.5,x[1]+0.2
    if x[0]**2 + x[1]**2 <= 0.01:
        out = 0
    else:
        out = p[1] + p[2] + p[0] * x[0]**2 - p[1] * torch.cos(p[3] * 2 * torch.pi * x[0]) + x[1]**2 - p[2] * torch.cos(p[4] * 2 * torch.pi * x[1])

    return out

def f2(x, p=None): 
    # Function F4
    # Default parameter values if 'p' is not provided: [20, 1, 1, 1, 1, 0.5, 0.5]

    if p is None:
        p = torch.tensor([20, 1, 1, 1, 1, 0.5, 0.5], dtype=torch.float32)
    # x[0],x[1]=x[0]+0.5,x[1]+0.2
    if x[0]**2 + x[1]**2 <= 0.01:
        out = 0
    else:
        out = -p[0] * torch.exp(-0.2 * torch.sqrt(p[5] * (p[1] * x[0]**2 + p[2] * x[1]**2))) - torch.exp(p[6] * (torch.cos(p[3] * 2 * torch.pi * x[0]) + torch.cos(p[4] * 2 * torch.pi * x[1]))) + p[0] + torch.exp(torch.tensor(1))

    return out


def f3(x, p=None):  
    # Default parameter values
    if p is None:
        p = torch.tensor([1, 1, 1, 1, 0.95], dtype=torch.float64)

    if x[0].item()**2 + x[1].item()**2 <= 0.01:
        out = torch.tensor(0.0, dtype=torch.float64)
    else:
        out = -torch.cos(p[2] * x[0] + math.pi) * torch.cos(p[3] * x[1] + math.pi) * \
              torch.exp(-p[0] * x[0]**2 - p[1] * x[1]**2) + 1
        out = torch.min(out, p[4])

    return out

def f4(x, p=None): 
    # Default parameter values
    if p is None:
        p = torch.tensor([1, 1, 0.3, 0.4], dtype=torch.float64)

    if x[0].item()**2 + x[1].item()**2 <= 0.01:
        outputArg1 = torch.tensor(0.0, dtype=torch.float64)
    else:
        outputArg1 = p[0]*x[0]**2 + p[1]*x[1]**2 - p[2]*torch.cos(10*p[2]*math.pi*x[0]) - \
                      p[3]*torch.cos(10*p[3]*math.pi*x[1]) + p[2] + p[3]

    return outputArg1

def f14(x, p=None):
    # Default parameter values  22.2015
    if p is None:
        p = torch.tensor([20, 0.2, 7, 7, 0.5], dtype=torch.float64)

    if x[0].item()**2 + x[1].item()**2 <= 0.01:
        out = torch.tensor(0.0, dtype=torch.float64)
    else:
        out = -p[0]*torch.exp(-p[1]*torch.sqrt(0.5*((p[2]*x[0])**2 + (p[3]*x[1])**2))) - \
              torch.exp(p[4]*(torch.cos(2*math.pi*(p[2]*x[0])) + torch.cos(2*math.pi*(p[3]*x[1])))) + p[0] + torch.exp(torch.tensor(1))

    return out

def f15(x, p=None):
    # Default parameter values  103.0457
    if p is None:
        p = torch.tensor([200, 0.02, 32/5, 32/5], dtype=torch.float64)

    if x[0].item()**2 + x[1].item()**2 <= 0.01:
        outputArg1 = torch.tensor(0.0, dtype=torch.float64)
    else:
        outputArg1 = -p[0]*torch.exp(-p[1]*torch.sqrt((p[2]*x[0])**2 + (p[3]*x[1])**2)) + p[0]

    return outputArg1

def f16(x, p=None):
    # Default parameter values 6.3615
    if p is None:
        p = torch.tensor([1, 1, 1, 1], dtype=torch.float64)

    if x[0].item()**2 + x[1].item()**2 <= 0.01:
        outputArg1 = torch.tensor(0.0, dtype=torch.float64)
    else:
        outputArg1 = (p[0]*torch.abs(x[0]) + p[1]*torch.abs(x[1])) * \
                      torch.exp(-p[2]*torch.sin(x[0])**2 - p[3]*torch.sin(x[1])**2)

    return outputArg1

def f21(x, p=None):
    # Default parameter values 1.2800
    if p is None:
        p = torch.tensor([0.1, 1, 1, 0.2, 0.2], dtype=torch.float64)

    if x[0].item()**2 + x[1].item()**2 <= 0.01:
        outputArg1 = torch.tensor(0.0, dtype=torch.float64)
    else:
        outputArg1 = -p[0]*(torch.cos(p[1]*math.pi*x[0]) + torch.cos(p[2]*math.pi*x[1])) + \
                      (x[0]*p[3])**2 + (x[1]*p[4])**2 + p[0]*2

    return outputArg1

def f22(x, p=None):
    # Default parameter values
    if p is None:
        p = torch.tensor([1, 1, 25], dtype=torch.float64)

    if x[0].item()**2 + x[1].item()**2 <= 0.01:
        outputArg1 = torch.tensor(0.0, dtype=torch.float64)
    else:
        outputArg1 = p[0]*x[0]**2 + p[1]*x[1]**2 + p[2]*(torch.sin(p[0]*x[0])**2 + torch.sin(p[1]*x[1])**2)

    return outputArg1

