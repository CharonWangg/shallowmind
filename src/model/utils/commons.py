import re
import torch
import torch.nn as nn


def add_prefix(prefix, name_dict, seperator='_'):
    # add prefix to the keys of a dict
    return {prefix+seperator+key: value for key, value in name_dict.items()}


def pascal_case_to_snake_case(camel_case):
    # convert PascalCase to snake_case
    # if input like 'AUROC':
    if camel_case.isupper():
        return camel_case.lower()
    else:
        snake_case = re.sub(r"(?P<key>[A-Z])", r"_\g<key>", camel_case)
        return snake_case.lower().strip('_')


def snake_case_to_pascal_case(snake_case):
    # convert snake_case to PascalCase
    if snake_case.islower():
        return snake_case.title()
    else:
        words = snake_case.split('_')
        return ''.join(word.title() for word in words)


def build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size,
                     norm_cfg='BatchNorm2d', act_cfg='ReLU', stride=1, padding=0, bias=True):
    # build convolution layer with norm and activation
    conv = getattr(nn, conv_cfg)
    act = getattr(nn, act_cfg)
    norm = getattr(nn, norm_cfg)
    conv_layer = nn.Sequential(
        conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        norm(num_features=out_channels),
        act(inplace=True)
    )
    return conv_layer


def infer_output_shape(model, tensor, flatten=False):
    # infer the output shape of a model by a given input tensor
    output = model(tensor)
    if isinstance(output, list):
        output_shape = [torch.numel(out) for out in output] if flatten else [out.shape for out in output]
    else:
        output_shape = torch.numel(output) if flatten else output.shape

    return output_shape


def magic_replace(model, magic_replacement):
    # use the given string to replace a specific layer in a model
    if magic_replacement and isinstance(magic_replacement, list) and len(magic_replacement) > 0:
        if isinstance(magic_replacement[0], tuple):
            for layer_name, layer in magic_replacement:
                # replace a layer with a specific layer
                # prevent the script executing weird things
                assert 'nn.' in magic_replacement[1], 'magic replacement must be a nn layer'
                model.__dict__[layer_name] = eval(layer)
        elif isinstance(magic_replacement[0], str) and isinstance(magic_replacement[1], str) and \
                len(magic_replacement) == 2:
            assert 'nn.' in magic_replacement[1], 'magic replacement must be a nn layer'
            model.__dict__[magic_replacement[0]] = eval(magic_replacement[1])
        else:
            raise ValueError('magic replacement must be a list of tuple or a tuple')
    return model


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class ResidualConcat(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        return torch.cat([x, res], dim=-1)