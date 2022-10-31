import torch
import pytorch_lightning as pl
from ..utils import add_prefix, infer_output_shape
from ..builder import ARCHS
from ...data.pipeline import Compose


@ARCHS.register_module()
class BaseArch(pl.LightningModule):
    def __init__(self, dataloader=None, pipeline=None, **kwargs):
        super(BaseArch, self).__init__(**kwargs)
        # build dataloader and pipeline (if needed)
        if dataloader is not None:
            self.dataloader = dataloader
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

    def exact_feat(self, x):
        x = self.backbone(x)
        return x

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
        if isinstance(label, dict):
            # multi-label multi-loss learning
            for label_type, label_value in label.items():
                if 'main' in label_type:
                    loss.update(self.forward_decode_train(feat, label_value))
                if 'aux' in label_type:
                    loss.update(self.forward_auxiliary_train(feat, label_value))
        else:
            # single-label multi-loss learning
            loss.update(self.forward_decode_train(feat, label))
            loss.update(self.forward_auxiliary_train(feat, label))

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

    def pipeline_model(self):
        if getattr(self, 'pipeline', None) is not None:
            self.model = self.pipeline(self.model)

    def pop_dataloader(self):
        # pop out dataloader
        if getattr(self, 'dataloader', None) is not None:
            self.__dict__.pop('dataloader')

    def pop_pipeline(self):
        # pop out pipeline
        if getattr(self, 'pipeline', None) is not None:
            self.__dict__.pop('pipeline')

    def cleanup(self):
        # pop out things don't need to be saved
        self.pipeline_model()
        self.pop_dataloader()
        self.pop_pipeline()

    def infer_input_shape_for_head(self, head, flatten=True):
        if head.get('in_channels', None) is None:
            if getattr(self, 'dataloader', None) is not None:
                example = next(iter(self.dataloader))[0]
                if isinstance(example, torch.Tensor):
                    tensor = example[0].unsqueeze(0)
                elif isinstance(example, dict):
                    assert 'seq' or 'image' in example.keys(), "example must contain 'seq' or 'image' key"
                    data_key = 'seq' if 'seq' in example else 'image'
                    tensor = example[data_key][0].unsqueeze(0)
                else:
                    raise TypeError('example must be a dict or a torch.Tensor')
                # auto infer the in_channels of head by the output shape of backbone
                in_index = head.get('in_index', -1)
                input_shape = infer_output_shape(self.backbone, tensor, flatten=flatten)
                in_channels = input_shape[in_index]
            else:
                raise ValueError('auto inference of output shape must be used with "need_dataloader=True"')
        else:
            in_channels = head.in_channels
        return in_channels