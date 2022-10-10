# ShallowMind
### A general ANN framework inspired by MMSegmentation

---------------------------------------------------------------------------
Inspired by MMSegmentation, I want all of my Deep Learning projects to be
simplified to a single 'config.py', and do not need to worried about training 
except for specific model/dataset. So I decide to make a highly disentangled 
training framework based on Pytorch-lightning.

Now support:
  * Custom Model Class
    * Architecture like BaseEncoderDecoder, NeuralEncoders, 
    * Backbone, 
      * Embedding layers like Base/Linear/Convolutional
      * LSTM/Transformer/TCN/Timm_models/NeuralPredictors
    * Head like MLP, Poolers
  * Custom Dataset Class
    * Single dataset and Multiple datasets concatenation
    * Pipeline for augmentations from Albumentations, Tsaug, etc
  * Various Optimizers and Schedulers 
    * Warmup like Linear, Cosine
    * Optimizers like Adam, AdamW, SGD, etc
    * Schedulers like OneCycleLR, CosineAnnealingLR, etc
  * Various Loss and Metrics
    * Multi loss fusion like CrossEntropy, BCE, FocalLoss, etc
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

Will expand it with my own projects (Next probably Huggingface series?), and welcome to contribute your model/dataset!

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
### Example
* [ResNet50 on CIFAR10](configs/image_classification_example.py) 94.11% Top1 Acc
