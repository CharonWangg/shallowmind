# ShallowMind
A self-use config-based training tool for deep learning
---------------------------------------------------------------------------
Amazed and inspired by [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), 
I want my Deep Learning project could be simplified to a single 'config.py' 
(probably a specific dataset/model class), and do not need to worried about training. 
So I decide to make a config-based training tool based on Pytorch-lightning with disentangle modules.

Still in the development stage...

---------------------------------------------------------------------------
### Installation
python >= 3.9
```
$ git clone https://github.com/CharonWangg/shallowmind/tree/light run
$ cd shallowmind && pip install -e .
```
### Usage
Demo image classification task on CIFAR10 with the ResNet50 backbone from Timm Models
```
$ cd shallowmind && python api/train.py --config configs/image_classification_example.py
```
Function used to load the trained checkpoint
```
from shallomwind.api.infer import prepare_inference
# di: correponding datainterface object in the config.py
# mi: correponding model object in the config.py (loaded checkpoint weight)
di, mi = prepare_inference(config_path, checkpoint_path)
```
Then you can either use trainer from pytorch-lightning to do following things on checkpoints
```
from pytorch_lightning import Trainer
trainer = Trainer(gpus=1)
trainer.test(mi, di.test_dataloader())
```
or write a naive inference loop
```
for batch in di.test_dataloader():
    mi.eval()
    with torch.no_grad():
        output = mi(batch)
```
