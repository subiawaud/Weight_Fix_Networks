# Weight Fixing Networks

<img src="https://user-images.githubusercontent.com/13983188/177870968-d26c4c87-6259-493a-b67e-8ccbc82dccbf.png" width="700">

This repo contains the Pytorch + Lightning code to apply the method proposed in 'Weight Fixing Networks' an accepted paper in ECCV 2022.



<img src="https://user-images.githubusercontent.com/13983188/177872026-dc25192e-f218-4b98-a68d-26332d6d47b9.png" width="700">



# Model Saves

Below we link to the quantised model saves quoted in the results section of the paper. 

| Model |  δ | Unique Param Count | Entropy | Acc | Link |
| ----- | --- | --- | --- | --- | --- | 
| ResNet-18 | 0.0075 | 193 | 4.15 | 70.3 | [link](https://drive.google.com/file/d/1xgoenu43v4tAGV4deSzRSjsQQQKTDeWg/view?usp=sharing)  |
| ResNet-18 | 0.01 | 164 |3.01 | 69.7 | [link](https://drive.google.com/file/d/1jPfjC0wIqpyTUDt8_hVIzSLFpz-_8Wwl/view?usp=sharing) |
| ResNet-18 | 0.015 | 90 | 2.72 | 67.3 | [link](https://drive.google.com/file/d/1xSMRcacasoXLaUj_6z0vfM9DDM0S8chs/view?usp=sharing) |
| ResNet-34 | 0.0075 | 233 | 3.87 | 73.0 | [link](https://drive.google.com/file/d/1MEq2oxT1t-3e8KrUDRFt8SgiKIuGE-TU/view?usp=sharing)  |
| ResNet-34 | 0.01 | 164 | 3.48 | 72.6 | [link](https://drive.google.com/file/d/1sTsaK__5qWdND-MTHZQf0xgRcaaFNIGL/view?usp=sharing) |
| ResNet-34 | 0.015 | 117 | 2.83 | 72.2 | [link](https://drive.google.com/file/d/18Q3jAqA2EMNHdt0tP2I7jpy8VCptr6mR/view?usp=sharing) |
| ResNet-50 | 0.0075 | 261| 4.11 | 76.0 | [link](https://drive.google.com/file/d/1c5PK_V_C8kd-fmQg4yf3SE_nj5uomRpi/view?usp=sharing) |
| ResNet-50 | 0.01 | 199 | 4.00 | 75.4 | [link](https://drive.google.com/file/d/1c5PK_V_C8kd-fmQg4yf3SE_nj5uomRpi/view?usp=sharing) |
| ResNet-50 | 0.015 | 125| 3.55 | 75.1 | [link](https://drive.google.com/file/d/1wDMzEeGb0kLCMVTfB35mAAELbOSYUOV4/view?usp=sharing) |

# To Run 

1. First make sure you have installed the requirements found in [requirements.txt](requirements.txt)

2. If you want to run the ImageNet experiments, you'll need to update the data_dir (see - [Setting ImageNet file locations](#setting-imageNet-file-locations)) 
3. Now just run `python pretrained_model_experiments.py` with any options arguments you wish to change from: 

Optional arguments: 


> ![#c5f015](https://via.placeholder.com/15/c5f015/c5f015.png) --distance_allowed : δ in the paper

> ![#c5f015](https://via.placeholder.com/15/c5f015/c5f015.png) --percentages : the percentage of weights clustered in each iteration 

> ![#c5f015](https://via.placeholder.com/15/c5f015/c5f015.png) --optimizer : training optimizer 

> ![#c5f015](https://via.placeholder.com/15/c5f015/c5f015.png) --experiment_name: used to save tb logs and model saves 

> ![#c5f015](https://via.placeholder.com/15/c5f015/c5f015.png) --scheduler : the learning rate scheduler 

> ![#c5f015](https://via.placeholder.com/15/c5f015/c5f015.png) -- lr : the learning rate

> ![#c5f015](https://via.placeholder.com/15/c5f015/c5f015.png) --first_epoch : the number of training iterations before any clustering (set to zero for pre-trained models) 

> ![#c5f015](https://via.placeholder.com/15/c5f015/c5f015.png) --fixing_epochs : the number of training epochs within a single clustering iteration (3 was used in the paper) 

> ![#c5f015](https://via.placeholder.com/15/c5f015/c5f015.png) --model : the name of the model to train see [get_model()](pre_trained_model_experiments.py) for a list of out-the-box supported dataset-model combinations

> ![#c5f015](https://via.placeholder.com/15/c5f015/c5f015.png) --dataset : the dataset to train on, currently we support CIFAR-10 and Imagenet 

> ![#c5f015](https://via.placeholder.com/15/c5f015/c5f015.png) --zero_distance : $\gamma_0$ in the paper, any abs weight less than this will be set to zero and prunned 

> ![#c5f015](https://via.placeholder.com/15/c5f015/c5f015.png) --regularisation_ratio : the weighting of the $\mathcal{L}_{reg}$ term

> ![#c5f015](https://via.placeholder.com/15/c5f015/c5f015.png) --bn_inc : whether to quantize batch-norm layers 

> ![#c5f015](https://via.placeholder.com/15/c5f015/c5f015.png) --resume : if continuing training, set this to the iteration you wish to continue from 

> ![#c5f015](https://via.placeholder.com/15/c5f015/c5f015.png) --calculation_type: the distance type you want to use, relative and euclidean supported 


#  Applying WFN to Different Types of Models 

To add new models for WFN quantisation, go to the get_model function within [pre_trained_model_experiments.py](pre_trained_model_experiments.py) and follow the format. Once added here, the new model will automatically be converted in a weight_fix_base which contains all the functionality needed to apply the clustering.

```python 

def get_model(model_name, data):
    """ Here is where the models are defined, if you would like to use a new model, you can insert it into here """

    if model_name == 'conv4':
        model = All_Conv_4()
        model = model.load_from_checkpoint(checkpoint_path="Pretrained_Models/PyTorch_CIFAR10/cifar10_models/state_dicts/all_conv4")

    if model_name == 'resnet18' and data == 'cifar10':
        model = resnet18(pretrained=True)

    if model_name == 'resnet34' and data == 'imnet':
        model = models.resnet34(pretrained=True)
```

# Setting ImageNet File Locations 


Go to the ImageNet module within [Datasets/imagenet.py](Datasets/imagenet.py) and edit the line self.data_dir to point to your imagenet data directory. 

```python
class ImageNet_Module(pl.LightningDataModule):
        def __init__(self, data_dir = 'Datasets/', shuffle_pixels=False, shuffle_labels=False, random_pixels=False):
                super().__init__()
                self.data_dir = 'Your data directory here'
                self.mean = [0.485, 0.456, 0.406]
                self.std = [0.229, 0.224, 0.225]
                self.normalise = transforms.Normalize(mean=self.mean, std=self.std)
                self.transform = self.transform_select(shuffle_pixels, random_pixels)
                self.test_trans = self.test_transform()
                self.target_transform = self.target_transform_select(shuffle_labels)
                self.targets = 1000
                self.dims = (3,224,224)
                self.bs = 64
```



