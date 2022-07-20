import torch
import torch.nn as nn
import pytorch_lightning as pl
from ..utils import add_prefix
from ..builder import ARCHS
from ..builder import build_backbone, build_head
from neuralpredictors.training.context_managers import eval_state
import copy

def prepare_grid(grid_mean_predictor, dataloaders):
    """
    Utility function for using the neurons cortical coordinates
    to guide the readout locations in image space.

    Args:
        grid_mean_predictor (dict): config dictionary, for example:
          {'type': 'cortex',
           'input_dimensions': 2,
           'hidden_layers': 1,
           'hidden_features': 30,
           'final_tanh': True}

        dataloaders: a dictionary of dataloaders, one PyTorch DataLoader per session
            in the format {'data_key': dataloader object, .. }
    Returns:
        grid_mean_predictor (dict): config dictionary
        grid_mean_predictor_type (str): type of the information that is being used for
            the grid positition estimator
        source_grids (dict): a grid of points for each data_key

    """
    if grid_mean_predictor is None:
        grid_mean_predictor_type = None
        source_grids = None
    else:
        grid_mean_predictor = copy.deepcopy(grid_mean_predictor)
        grid_mean_predictor_type = grid_mean_predictor.pop("type")

        if grid_mean_predictor_type == "cortex":
            input_dim = grid_mean_predictor.pop("input_dimensions", 2)
            source_grids = {
                k: v.dataset.dataset.neurons.cell_motor_coordinates[:, :input_dim]
                for k, v in dataloaders.items()
            }
    return grid_mean_predictor, grid_mean_predictor_type, source_grids

def get_module_output(model, input_shape, use_cuda=True):
    """
    Return the output shape of the model when fed in an array of `input_shape`.
    Note that a zero array of shape `input_shape` is fed into the model and the
    shape of the output of the model is returned.

    Args:
        model (nn.Module): PyTorch module for which to compute the output shape
        input_shape (tuple): Shape specification for the input array into the model
        use_cuda (bool, optional): If True, model will be evaluated on CUDA if available. Othewrise
            model evaluation will take place on CPU. Defaults to True.

    Returns:
        tuple: output shape of the model

    """
    # infer the original device
    initial_device = next(iter(model.parameters())).device
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
    with eval_state(model):
        with torch.no_grad():
            input = torch.zeros(1, *input_shape[1:], device=device)
            output = model.to(device)(input)
    model.to(initial_device)
    return output[-1].shape

@ARCHS.register_module()
class FiringRateEncoder(pl.LightningModule):
    def __init__(self, backbone, head, auxiliary_head=None, dataloader=None):
        super(FiringRateEncoder, self).__init__()
        if dataloader is None:
            raise ValueError('dataloader is required for initializing FiringRateEncoder')

        # ****************************Modified from official code******************************************
        # Obtain the named tuple fields from the first entry of the first dataloader in the dictionary
        subject = dataloader.dataset.subject
        in_name, out_name = "images", "responses"
        session_shape_dict = {k: v.shape for k, v in next(iter(dataloader))[0].items()}
        n_neurons_dict = {subject: session_shape_dict[out_name][1]}
        input_channels = [session_shape_dict[in_name][1]]

        core_input_channels = (
            list(input_channels.values())[0]
            if isinstance(input_channels, dict)
            else input_channels[0]
        )

        dataloader = {subject: dataloader}
        grid_mean_predictor, grid_mean_predictor_type, source_grids = prepare_grid(head.grid_mean_predictor, dataloader)

        backbone.input_channels = core_input_channels
        self.backbone = build_backbone(backbone)

        in_shapes_dict = {
            subject: get_module_output(self.backbone, session_shape_dict[in_name])[1:]
        }

        head.in_shape_dict = in_shapes_dict
        head.n_neurons_dict = n_neurons_dict
        head.loader = dataloader
        head.grid_mean_predictor_type = grid_mean_predictor_type
        head.grid_mean_predictor = grid_mean_predictor
        head.source_grids = source_grids
        self.head = build_head(head)
        self.auxiliary_head = None
        # ****************************Modified from official code******************************************

    def exact_feat(self, x):
        x = x['images']
        x = self.backbone(x)
        return x

    def regularizer(self):
        regularization = torch.zeros(1, device=self.device)
        if getattr(self.backbone.model, 'regularizer'):
            regularization += self.backbone.model.regularizer()
        if getattr(self.head.model, 'regularizer'):
            regularization += self.head.model.regularizer()
        return regularization

    def forward_decode_train(self, feat, label):
        loss = dict()
        decode_loss = self.head.forward_train(feat, label)
        loss.update(add_prefix(f'mainhead', decode_loss))
        return loss

    def forward_auxiliary_train(self, feat, label):
        loss = dict()
        if self.auxiliary_head is not None:
            for idx, auxiliary_head in enumerate(self.auxiliary_head):
                loss.update(add_prefix(f'auxhead{idx}', auxiliary_head.forward_train(feat, label)))
        return loss

    def forward_train(self, x, label):
        loss = dict()
        feat = self.exact_feat(x)

        loss.update(self.forward_decode_train(feat, label))
        loss.update(self.forward_auxiliary_train(feat, label))

        # add regularization
        loss.update({'regularization_loss': self.regularizer()})
        # sum up all losses
        loss.update({'loss': sum([loss[k] for k in loss.keys() if 'loss' in k.lower()])})

        # pack the output and losses
        return loss

    def forward_test(self, x, label=None):
        feat = self.exact_feat(x)
        res = self.head.forward_test(feat, label)

        # sum up all losses
        if label is not None:
            res.update({'loss': sum([res[k] for k in res.keys() if 'loss' in k.lower()])})
        else:
            res.update({'loss': 'Not available'})
        return res
