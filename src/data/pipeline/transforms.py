import cv2
import copy
import numpy as np
import albumentations
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
import neuralpredictors.data.transforms as neural_transforms
from shallowmind.src.data.builder import build_pipeline, PIPELINES

@PIPELINES.register_module()
class LoadImages(object):
    def __init__(self, channels_first=False, to_RGB=True, **kwargs):
        self.channels_first = channels_first
        self.to_RGB = to_RGB
        self.kwargs = kwargs

    def __call__(self, data):
        image = data["image"]
        # get the channel dimension
        channel_dim = np.argmin(image.shape)
        if self.channels_first:
            if channel_dim != 0:
                if self.to_RGB:
                    if image.shape[channel_dim] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    elif image.shape[channel_dim] == 1:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                image = image.transpose((2, 0, 1))
        else:
            if channel_dim != 2:
                image = image.transpose((1, 2, 0))
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[-1] == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        data["image"] = image
        return data


@PIPELINES.register_module()
class ToTensor:
    def __init__(self, transpose_mask=False, always_apply=True, p=1.0):
        self.totensorv2 = ToTensorV2(transpose_mask=transpose_mask,  always_apply=always_apply, p=p)

    def __call__(self, data):
        data['image'] = self.totensorv2(image=data['image'])['image']
        return data


# TODO: Refine this function
@PIPELINES.register_module()
class ImageSelection:
    '''Image selection based on img_ids/img_conditions/img_n'''
    def __init__(self, dat, image_ids=None, image_condition=None, image_n=None):
        self.__dict__.update(locals())
        dat_info = dat.info if not file_tree else dat.trial_info
        if "image_id" in dir(dat_info):
            frame_image_id = dat_info.image_id
            image_class = dat_info.image_class
        elif "colorframeprojector_image_id" in dir(dat_info):
            frame_image_id = dat_info.colorframeprojector_image_id
            image_class = dat_info.colorframeprojector_image_class
        elif "frame_image_id" in dir(dat_info):
            frame_image_id = dat_info.frame_image_id
            image_class = dat_info.frame_image_class
        else:
            raise ValueError(
                "'image_id' 'colorframeprojector_image_id', or 'frame_image_id' have to present in the dataset under dat.info "
                "in order to load get the oracle repeats."
            )

        if isinstance(image_condition, str):
            image_condition_filter = image_class == image_condition
        elif isinstance(image_condition, list):
            image_condition_filter = sum(
                [image_class == i for i in image_condition]
            ).astype(np.bool)
        else:
            if image_condition is not None:
                raise TypeError(
                    "image_condition argument has to be a string or list of strings"
                )
        self.image_id_array = frame_image_id
        self.tier

    def __call__(self):
        # sample images
        if self.tier == "train" and self.image_ids is not None and self.image_condition is None:
            subset_idx = [
                np.where(self.image_id_array == image_id)[0][0] for image_id in self.image_ids
            ]
            assert (
                    sum(self.tier_array[subset_idx] != "train") == 0
            ), "image_ids contain validation or test images"
        elif self.tier == "train" and self.image_n is not None and self.image_condition is None:
            random_state = np.random.get_state()
            if self.image_base_seed is not None:
                np.random.seed(
                    self.image_base_seed * self.image_n
                )  # avoid nesting by making seed dependent on number of images
            subset_idx = np.random.choice(
                np.where(self.tier_array == "train")[0], size=self.image_n, replace=False
            )
            np.random.set_state(random_state)
        elif self.image_condition is not None and self.image_ids is None:
            subset_idx = np.where(
                np.logical_and(self.image_condition_filter, tier_array == self.tier)
            )[0]
            assert (
                    sum(self.tier_array[subset_idx] != tier) == 0
            ), "image_ids contain validation or test images"
        else:
            subset_idx = np.where(self.tier_array == tier)[0]

@PIPELINES.register_module()
class NeuronSelection:
    '''Neural Selection based on areas/layers/neuron_n/neuron_ids'''

    def __init__(self, areas=None, layers=None, neuron_n=None,
                 neuron_base_seed=None, neuron_ids=None, exclude_neuron_n=0):
        self.__dict__.update(locals())
        assert any(
            [
                neuron_ids is None,
                all(
                    [
                        neuron_n is None,
                        neuron_base_seed is None,
                        areas is None,
                        layers is None,
                        exclude_neuron_n == 0,
                    ]
                ),
            ]
        ), "neuron_ids can not be set at the same time with any other neuron selection criteria"
        assert any(
            [exclude_neuron_n == 0, neuron_base_seed is not None]
        ), "neuron_base_seed must be set when exclude_neuron_n is not 0"

    def __call__(self, data):
        dat = copy.deepcopy(data)
        conds = np.ones(len(dat.neurons.area), dtype=bool)
        if self.areas is not None:
            conds &= np.isin(dat.neurons.area, self.areas)
        if self.layers is not None:
            conds &= np.isin(dat.neurons.layer, self.layers)
        idx = np.where(conds)[0]
        if self.neuron_n is not None:
            random_state = np.random.get_state()
            if self.neuron_base_seed is not None:
                np.random.seed(
                    self.neuron_base_seed * self.neuron_n
                )  # avoid nesting by making seed dependent on number of neurons
            assert (
                    len(dat.neurons.unit_ids) >= self.exclude_neuron_n + self.neuron_n
            ), "After excluding {} neurons, there are not {} neurons left".format(
                self.exclude_neuron_n, self.neuron_n
            )
            neuron_ids = np.random.choice(
                dat.neurons.unit_ids, size=self.exclude_neuron_n + self.neuron_n, replace=False
            )[self.exclude_neuron_n:]
            np.random.set_state(random_state)
        if neuron_ids is not None:
            idx = [
                np.where(dat.neurons.unit_ids == unit_id)[0][0] for unit_id in neuron_ids
            ]
        if idx is not None:
            dat['sample_idx'] = idx
        return dat

@PIPELINES.register_module()
class NeuralPredictors:
    '''Neural Predictors augmentations'''

    def __init__(self, transforms):

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        self.transforms = transforms

        self.aug = []
        for t in self.transforms:
            self.aug.append(self.neural_builder(t))

    def neural_builder(self, cfg):
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            if neural_transforms is None:
                raise RuntimeError('neuralpredictors is not installed')
            obj_cls = getattr(neural_transforms, obj_type)
        else:
            raise TypeError(
                f'type must be a str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.neural_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    def __call__(self, results):
        # dict to albumentations format, temporarily no segmentation task, so only image aug (without mask)
        res = copy.deepcopy(results)
        try:
            for aug in self.aug:
                aug_res = aug(image=results['image'])
                results['image'] = aug_res['image']
                return results
        except Exception as e:
            print(e)
            return res

@PIPELINES.register_module()
class Albumentations:
    '''Albumentation augmentation'''

    def __init__(self, transforms):
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        self.transforms = transforms

        self.aug = []
        for t in self.transforms:
            self.aug.append(self.albu_builder(t))
        self.aug = Compose(self.aug)

    def albu_builder(self, cfg):
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        else:
            raise TypeError(
                f'type must be a str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)


    def __call__(self, results):
        # dict to albumentations format, temporarily no segmentation task, so only image aug (without mask)
        res = copy.deepcopy(results)
        try:
            aug_res = self.aug(image = res['image'])
            res['image'] = aug_res['image']
            return res
        except Exception as e:
            print(e)
            return res

