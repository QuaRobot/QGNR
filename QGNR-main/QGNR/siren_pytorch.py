import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import torchquantum as tq


class QuantumLayer(nn.Module):
    def __init__(self, n_wires, n_blocks):
        super().__init__()
        self.n_wires = n_wires
        self.n_blocks = n_blocks
        self.measurez = tq.MeasureAll(tq.PauliZ)

        encoder = []
        index = 0
        for i in range(n_wires):

            encoder.append({'input_idx': [index], 'func': 'rz', 'wires': [i]})
            index += 1


        self.encoder = tq.GeneralEncoder(encoder)
        self.rx1_layers = tq.QuantumModuleList()
        self.ry1_layers = tq.QuantumModuleList()
        self.rz1_layers = tq.QuantumModuleList()
        self.cnot_layers = tq.QuantumModuleList()
        self.rx2_layers = tq.QuantumModuleList()
        self.ry2_layers = tq.QuantumModuleList()
        self.rz2_layers = tq.QuantumModuleList()

        for _ in range(n_blocks):
            self.rx1_layers.append(
                tq.Op1QAllLayer(
                    op=tq.RZ,
                    n_wires=n_wires,
                    has_params=True,
                    trainable=True,
                )
            )
            self.ry1_layers.append(
                tq.Op1QAllLayer(
                    op=tq.RY,
                    n_wires=n_wires,
                    has_params=True,
                    trainable=True,
                )
            )
            self.rz1_layers.append(
                tq.Op1QAllLayer(
                    op=tq.RZ,
                    n_wires=n_wires,
                    has_params=True,
                    trainable=True,
                )
            )

            self.cnot_layers.append(
                tq.Op2QAllLayer(
                    op=tq.CNOT,
                    n_wires=n_wires,
                    has_params=False,
                    trainable=False,
                    circular=True,
                )
            )

            self.rx2_layers.append(
                tq.Op1QAllLayer(
                    op=tq.RZ,
                    n_wires=n_wires,
                    has_params=True,
                    trainable=True,
                )
            )
            self.ry2_layers.append(
                tq.Op1QAllLayer(
                    op=tq.RY,
                    n_wires=n_wires,
                    has_params=True,
                    trainable=True,
                )
            )
            self.rz2_layers.append(
                tq.Op1QAllLayer(
                    op=tq.RZ,
                    n_wires=n_wires,
                    has_params=True,
                    trainable=True,
                )
            )

    def forward(self, x):
        x = x.squeeze()
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        for k in range(self.n_blocks):
            self.rx1_layers[k](qdev)
            self.ry1_layers[k](qdev)
            self.rz1_layers[k](qdev)
            self.cnot_layers[k](qdev)
            self.rx2_layers[k](qdev)
            self.ry2_layers[k](qdev)
            self.rz2_layers[k](qdev)
            self.cnot_layers[k](qdev)
            if k != self.n_blocks - 1:
                self.encoder(qdev, x)
        out = self.measurez(qdev).unsqueeze(0)
        return out



class HybridLayer(nn.Module):
    def __init__(self, in_features, hidden_features, spectrum_layer):
        super().__init__()
        self.clayer = nn.Linear(in_features, hidden_features)
        self.norm = nn.BatchNorm1d(hidden_features)
        self.qlayer1 = QuantumLayer(hidden_features, spectrum_layer)

    def forward(self, x):
        x = self.clayer(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.qlayer1(x)
        return x

class Hybridren(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, spectrum_layer):
        super().__init__()

        self.net = []
        self.net.append(HybridLayer(in_features, hidden_features, spectrum_layer))
        final_linear = nn.Linear(hidden_features, out_features)


        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = torch.unsqueeze(coords,dim=0)
        coords = coords.clone().detach().requires_grad_(True)
        output = nn.Sigmoid()(self.net(coords).squeeze(dim=0))
        return output



def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)


# siren layer
class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = 'sine'):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.act = activation

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None

       
        if activation=='sine':
            self.activation=Sine(w0)
        elif activation=='relu':
            self.activation=nn.ReLU(inplace=False)
        elif activation=='id':
            self.activation=nn.Identity()
        elif activation=='sigmoid':
            self.activation=nn.Sigmoid()
        else:
            raise ValueError('No mlp activation specified')

        #self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in
        act = self.act

        if act =='relu':
            w_std = math.sqrt(1/dim)
        else:
            w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        
        weight.uniform_(-w_std, w_std)
        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out



class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, activation = 'relu', final_activation = 'sigmoid'):
        super().__init__()

        self.dim_hidden = dim_hidden
        self.num_layers = len(dim_hidden)
        self.layers = nn.ModuleList([])
        for ind in range(self.num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden[ind-1]
            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden[ind],
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                activation = activation
            ))

        final_activation = 'id' if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden[num_layers-1], dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)


    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers)
        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x = x*rearrange(mod, 'd -> () d')

        return self.last_layer(x)





