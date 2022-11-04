# ShallowMind
A self-use config-based training tool for deep learning
---------------------------------------------------------------------------
Amazed and inspired by [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), 
I want my Deep Learning project could be simplified to a single 'config.py' 
(probably a specific dataset/model class), and do not need to worried about training. 
So I decide to make a config-based training tool based on Pytorch-lightning with disentangle modules.

Now support:
  * Custom Model Class
    * Base/General architecture like 
      * BaseEncoderDecoder (for Classification/Regression)
      * BaseVAE (for Recognition/Generation)
      * BaseGAN (for Generation)
      * BaseDQN (for Reinforcement Learning)
    * Specific architecture like FiringRateEncoder/NeuralEncoder/SLDiscoEncoder
    * Backbone, 
      * Embedding layers like Base/Linear/Convolutional/PositionalEncoding
      * MLP/BaseConvNet/LSTM/Transformer/TCN/Timm_models/NeuralPredictors
    * Head like MLP/Poolers/ConvTransposeHead
  * Custom Dataset Class
    * Support public dataset like TorchVision/Gym/NetSim/LiNGAM
    * Single dataset and Multiple datasets concatenation
    * Pipeline for augmentations from Albumentations, Tsaug, etc
  * Various Optimizers and Schedulers 
    * Warmup like Linear, Cosine
    * Optimizers like Adam, AdamW, SGD, etc (support multiple simultaneous optimizers)
    * Schedulers like OneCycleLR, CosineAnnealingLR, etc (support multiple simultaneous schedulers)
  * Various Loss and Metrics
    * Multi loss fusion like CrossEntropy, BCE, FocalLoss, etc (support weighted multi loss/auxiliary loss)
    * Multi metric mainly based on torchmetrics
  * Logging and Checkpointing
  * Distributed Training
  * Simple api train/infer for use

Feel free to combine the existed components to build your own model, or write
your special one.  
(E.x. BaseEncoderDecoder(Embedding(Conv)+Transformer+BaseHead) == ViT; 
      BaseEncoderDecoder(Timm_models+BaseHead) == Classic Image Classification Model;
      BaseEncoderDecoder(LSTM/Transformer/TCN+BaseHead)  == Sequence Prediction Model)
      ...
)

Will expand it with my own projects!


---------------------------------------------------------------------------
### Installation
python >= 3.9
```
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
### Example
* [ResNet50 on CIFAR10 in 100 epochs from a scratch](configs/resnet50_image_classification_example.py) 94.11% Top1 Acc
* [ViT on CIFAR10 in 300 epochs from a scratch](configs/vit_image_classification_example.py)
* [VAE on MNIST](configs/vae_image_mnist_example.py)
* [GAN on MNIST](configs/gan_image_mnist_example.py)
* [DQN on CartPole](configs/dqn_cartpole_example.py)
