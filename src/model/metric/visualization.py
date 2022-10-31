from copy import deepcopy
import torchvision
import pytorch_lightning as pl
from ..builder import METRICS
from ..utils import pascal_case_to_snake_case


@METRICS.register_module()
class ImageVisualization(pl.LightningModule):
    def __init__(self, n_sample=3, metric_name='ImageVisualization', image_name='generated_images', **kwargs):
        super(ImageVisualization, self).__init__()
        if metric_name is None:
            raise ValueError('metric_name is required')
        self.n_sample = n_sample
        self.image_name = image_name
        self.metric_name = pascal_case_to_snake_case(metric_name)

    def forward(self, pred, target=None):
        self.log_image(pred, name=self.image_name, index=self.trainer.global_step)
        return None

    def log_image(self, image, name="generated_images", index=0):
        image = image[:self.n_sample].detach().cpu()
        # log image to the logger
        for i in range(self.n_sample):
            self.trainer.logger.experiment.log_image(image_data=image[i], name=f'name={name}'+'-'+
                                                                               f'step={index}'+'-'+
                                                                               f'index={i}')

    def save_image(self, image, name="generated_images", index=0):
        raise NotImplementedError
