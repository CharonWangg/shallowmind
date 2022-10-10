import cv2
import copy
import torch
import numpy as np
import albumentations
from copy import deepcopy
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
import neuralpredictors.data.transforms as neural_transforms
import tsaug
import torchvision
from composer import functional as cf
from shallowmind.src.data.builder import build_pipeline, PIPELINES


@PIPELINES.register_module()
class LoadImages(object):
    def __init__(self, as_array=True, channels_first=False, to_RGB=True, **kwargs):
        self.channels_first = channels_first
        self.as_array = as_array
        self.to_RGB = to_RGB
        self.kwargs = kwargs

    def __call__(self, data):
        image = data["image"]
        if self.as_array:
            image = np.asarray(image)
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
                if self.to_RGB:
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
        for key, value in data.items():
            if key == "image":
                data['image'] = self.totensorv2(image=data['image'])['image']
            elif key == "seq":
                data[key] = torch.tensor(data[key], dtype=torch.float32)
            else:
                data[key] = torch.tensor(data[key])
        return data

@PIPELINES.register_module()
class CausalWindowCrop:
    PAD_IDX = -100
    def __init__(self, num_windows=32, window_size=512, prior_size=12, interval=10, random_start=True, eps=1e-6, p=1.0):
        self.num_windows = num_windows
        self.window_size = window_size
        self.prior_size = prior_size
        self.interval = interval
        self.random_start = random_start
        self.eps = eps
        self.p = p

    def __call__(self, data):
        data = deepcopy(data)
        x = data.pop('seq') # (L, C)
        feat_dim = x.shape[-1]
        x = np.diff(x[::self.interval, :], n=1, axis=0)
        # find spikes and their corresponding causal windows (NW, WL, C)
        spikes = np.sort(np.argsort(np.abs(x[:, 0]) ** 2)[::-1][:self.num_windows])
        # remove the zero-spikes
        spikes = [spike for spike in spikes if x[spike, 0] > self.eps]
        if not self.random_start:
            x = [x[(spike - self.prior_size):(spike + self.window_size - self.prior_size), :]
                  for spike in spikes
                  if spike + self.window_size - self.prior_size < x.shape[0] and spike - self.prior_size >= 0]
        else:
            spikes = [spike - self.prior_size - np.random.randint(0, self.window_size) for spike in spikes]
            spikes = [spike for spike in spikes if spike + self.window_size < x.shape[0] and spike - self.prior_size >= 0]
            x = [x[spike:(spike + self.window_size), :] for spike in spikes]
        if not x:
            data['seq'] = np.concatenate((np.zeros((1, self.window_size, feat_dim)),
                                     self.PAD_IDX * np.ones((self.num_windows - 1, self.window_size, feat_dim))), axis=0).astype(np.float32)
        else:
            x = np.stack(x, axis=0)[:self.num_windows]
            data['seq'] = np.concatenate((x, self.PAD_IDX * np.ones((self.num_windows - x.shape[0], self.window_size, feat_dim))), axis=0)
        data['padding_mask'] = (data['seq'][:, 0, 0] == self.PAD_IDX)

        return data

@PIPELINES.register_module()
class RandomWindowCrop:
    PAD_IDX = -100
    def __init__(self, num_windows=32, window_size=512, prior_size=12, interval=10, random_start=True, eps=1e-6, p=1.0):
        self.num_windows = num_windows
        self.window_size = window_size
        self.prior_size = prior_size
        self.interval = interval
        self.random_start = random_start
        self.eps = eps
        self.p = p

    def __call__(self, data):
        data = deepcopy(data)
        x = data.pop('seq') # (L, C)
        feat_dim = x.shape[-1]
        x = torch.diff(x[::self.interval, :], n=1, dim=0)
        # find spikes and their corresponding causal windows (NW, WL, C)
        spikes = torch.argsort(torch.abs(x[:, 0]) ** 2, descending=True)[:self.num_windows]
        # remove the zero-spikes
        spikes = [spike for spike in spikes if x[spike, 0] > self.eps]
        if not self.random_start:
            x = [x[(spike - self.prior_size):(spike + self.window_size - self.prior_size), :]
                  for spike in spikes
                  if spike + self.window_size - self.prior_size < x.shape[0] and spike - self.prior_size >= 0]
        else:
            spikes = torch.where(x[:, 0] == 1)[0].tolist() + torch.where(x[:, 0] == -1)[0].tolist()
            spikes = [spike - self.prior_size - np.random.randint(0, self.window_size) for spike in spikes]
            spikes = [spike for spike in spikes if spike + self.window_size < x.shape[0] and spike - self.prior_size >= 0]
            x = [x[spike:(spike + self.window_size), :] for spike in spikes]
        if not x:
            data['seq'] = torch.cat((torch.zeros((1, self.window_size, feat_dim)),
                                     self.PAD_IDX * torch.ones((self.num_windows - 1, self.window_size, feat_dim))), dim=0).to(torch.float32)
        else:
            x = torch.stack(x, dim=0)[:self.num_windows]
            data['seq'] = torch.cat((x, self.PAD_IDX * torch.ones((self.num_windows - x.shape[0], self.window_size, feat_dim))), dim=0)
        data['padding_mask'] = (data['seq'][:, 0, 0] == self.PAD_IDX)

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
class AddExtraFeatureAsChannels:
    '''Add extra feature as channels to the images'''
    def __init__(self, extra_feature_key):
        if isinstance(extra_feature_key, str):
            extra_feature_key = [extra_feature_key]
        elif not isinstance(extra_feature_key, list):
            print('extra_feature_key has to be a list')
        self.extra_feature_key = extra_feature_key

    def __call__(self, data):
        orig_feature = deepcopy(data['image'])
        for key in self.extra_feature_key:
            extra_feature = data[key]
            orig_feature = np.concatenate(
                (
                    orig_feature,
                    np.ones((1, *orig_feature.shape[-(len(orig_feature.shape) - 1) :]))
                    * np.expand_dims(extra_feature, axis=((len(orig_feature.shape) - 2), (len(orig_feature.shape) - 1))),
                ),
                axis=len(orig_feature.shape) - 3,
            )
        data['image'] = orig_feature.astype(np.float32)

        return data

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
        res = copy.deepcopy(results)
        try:
            kwargs = {}
            if res.get('mask', None) is not None:
                kwargs['mask'] = res['mask']
            if res.get('bbox', None) is not None:
                kwargs['bbox'] = res['bbox']
            if res.get('keypoints', None) is not None:
                kwargs['keypoints'] = res['keypoints']
            aug_res = self.aug(image=res['image'], **kwargs)
            # get image, mask, bbox, keypoint from aug_res
            res['image'] = aug_res['image']
            if aug_res.get('mask', None) is not None:
                res['mask'] = aug_res['mask']
            if aug_res.get('bbox', None) is not None:
                res['bbox'] = aug_res['bbox']
            if aug_res.get('keypoints', None) is not None:
                res['keypoint'] = aug_res['keypoint']
            return res
        except Exception as e:
            print(e)
            return res


@PIPELINES.register_module()
class TsAug:
    '''Time series augmentation'''

    def __init__(self, transforms):
        if tsaug is None:
            raise RuntimeError('tsaug is not installed')

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        self.transforms = transforms

        self.aug = []
        for t in self.transforms:
            self.aug.append(self.tsaug_builder(t))
        # for tsaug pipeline, use + instead of Compose
        self.aug = tsaug._augmenter.base._AugmenterPipe(self.aug)

    def tsaug_builder(self, cfg):
        """Import a module from tsaug.

        Extra parameters: frequency, .

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        frequency = args.pop('frequency', 1)
        probability = args.pop('p', 1.0)
        if isinstance(obj_type, str):
            if tsaug is None:
                raise RuntimeError('tsaug is not installed')
            obj_cls = getattr(tsaug, obj_type)
        else:
            raise TypeError(
                f'type must be a str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.tsaug_builder(transform)
                for transform in args['transforms']
            ]

        return (obj_cls(**args) * frequency ) @ probability


    def __call__(self, results):
        res = copy.deepcopy(results)
        try:
            mask = res.get('mask', None)
            res['seq'] = self.aug.augment(res['seq'], mask)
            return res
        except Exception as e:
            print(e)
            return res

@PIPELINES.register_module()
class Composer:
    '''Composer library for model pipeline'''

    def __init__(self, transforms):
        # if cf is None:
        #     raise RuntimeError('mosaicml is not installed')

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        self.transforms = transforms

    def algorithm_builder(self, cfg):
        """Import a module from composer.functional.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        func_type = 'apply_' + args.pop('type').lower()
        assert isinstance(func_type, str), f'type must be a str, but got {type(func_type)}'
        func = getattr(cf, func_type)

        return func, deepcopy(args)

    def __call__(self, results):
        res = copy.deepcopy(results)

        for t in self.transforms:
            func, kwargs = self.algorithm_builder(t)
            results = func(results, kwargs)

        return results

@PIPELINES.register_module()
class TorchVision:
    '''Torchvision library for image pipeline'''
    def __init__(self, transforms):
        if torchvision is None:
            raise RuntimeError('torchvision is not installed')

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        self.transforms = transforms

        self.aug = []
        for t in self.transforms:
            self.aug.append(self.torchvision_builder(t))
        self.aug = torchvision.transforms.Compose(self.aug)

    def torchvision_builder(self, cfg):
        """Import a module from torchvision.

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
            if torchvision is None:
                raise RuntimeError('torchvision is not installed')
            obj_cls = getattr(torchvision.transforms, obj_type)
        else:
            raise TypeError(
                f'type must be a str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.torchvision_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    def __call__(self, results):
        res = copy.deepcopy(results)
        try:
            aug_res = self.aug(res['image'])
            # get image, mask, bbox, keypoint from aug_res
            res['image'] = aug_res['image']
            return res
        except Exception as e:
            print(e)
            return res