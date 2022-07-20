from ..builder import build_pipeline

class Compose:
    '''Data preprocessing pipeline'''
    def __init__(self, transforms):
        if transforms is None:
            print('The pipeline is not setup, will use identity transform')
            self.transforms = []
        else:
            self.transforms = []
            self.pre_transforms = []
            for transform in transforms:
                if isinstance(transform, dict):
                    if transform.pop('pre_transform', False):
                        # transform to all data
                        transform = build_pipeline(transform)
                        self.pre_transforms.append(transform)
                    else:
                        # transform while iterations
                        transform = build_pipeline(transform)
                        self.transforms.append(transform)
                else:
                    raise TypeError('transform must be a dict')

    def pre_transform(self, data):
        for t in self.pre_transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data