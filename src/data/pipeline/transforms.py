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
    def __init__(self, as_array=True, channels_first=False, to_RGB=True, min_max_norm=True, **kwargs):
        self.channels_first = channels_first
        self.as_array = as_array
        self.to_RGB = to_RGB
        self.min_max_norm = min_max_norm
        self.kwargs = kwargs

    def __call__(self, data):
        image = data["image"]
        if self.as_array:
            image = np.asarray(image)
        # get the channel dimension
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=0)
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

        # min max normalization to [0, 1]
        if self.min_max_norm:
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min())

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
class NeuralPredictors:
    # Neural Predictors augmentations
    def __init__(self, transforms):
        transforms = copy.deepcopy(transforms)
        self.transforms = transforms

        self.aug = []
        for t in self.transforms:
            self.aug.append(self.neural_builder(t))

    def neural_builder(self, cfg):
        # Import a module from NeuralPredictors
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
    # Albumentation augmentation
    def __init__(self, transforms):
        if Compose is None:
            raise RuntimeError('albumentations is not installed')
        transforms = copy.deepcopy(transforms)
        self.transforms = transforms

        self.aug = []
        for t in self.transforms:
            self.aug.append(self.albu_builder(t))
        self.aug = Compose(self.aug)

    def albu_builder(self, cfg):
        # Import a module from albumentations
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
    # Time series augmentation
    def __init__(self, transforms):
        if tsaug is None:
            raise RuntimeError('tsaug is not installed')
        transforms = copy.deepcopy(transforms)
        self.transforms = transforms

        self.aug = []
        for t in self.transforms:
            self.aug.append(self.tsaug_builder(t))
        # for tsaug pipeline, use + instead of Compose
        self.aug = tsaug._augmenter.base._AugmenterPipe(self.aug)

    def tsaug_builder(self, cfg):
        # Import a module from tsaug
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
    # Composer library for model pipeline
    def __init__(self, transforms):

        transforms = copy.deepcopy(transforms)
        self.transforms = transforms

    def algorithm_builder(self, cfg):
        # Import a module from composer.functional
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
    # Torchvision library for image pipeline
    def __init__(self, transforms):
        if torchvision is None:
            raise RuntimeError('torchvision is not installed')
        transforms = copy.deepcopy(transforms)
        self.transforms = transforms

        self.aug = []
        for t in self.transforms:
            self.aug.append(self.torchvision_builder(t))
        self.aug = torchvision.transforms.Compose(self.aug)

    def torchvision_builder(self, cfg):
        # Import a module from torchvision
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