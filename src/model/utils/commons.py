import torch.nn as nn

def add_prefix(prefix, name_dict, seperator='_'):
    return {prefix+seperator+key: value for key, value in name_dict.items()}

def build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size,
                     norm_cfg='BatchNorm2d', act_cfg='ReLU', stride=1, padding=0, bias=True):
    conv = getattr(nn, conv_cfg)
    act = getattr(nn, act_cfg)
    norm = getattr(nn, norm_cfg)
    conv_layer = nn.Sequential(
        conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        norm(num_features=out_channels),
        act(inplace=True)
    )
    return conv_layer