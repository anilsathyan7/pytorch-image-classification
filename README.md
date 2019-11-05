## Pytorch-Image-Classification

A simple demo of **image classification** using pytorch. Here, we use a **custom dataset** containing **43956 images** belonging to **11 classes** for training(and validation). Also, we compare three different approaches for training viz. **training from scratch, finetuning the convnet and convnet as a feature extractor**, with the help of **pretrained** pytorch models. The models used include: **VGG11, Resnet18 and MobilenetV2**.

### Dependencies

* Python3, Scikit-learn
* Pytorch, PIL
* Torchsummary, Tensorboard

```python
pip install torchsummary # keras-summary
pip install tensorboard  # tensoflow-logging
```

**NB**: Update the libraries to their latest versions before training.

### How to run

Download the dataset: [imdb_small](https://drive.google.com/drive/folders/1xrHX-koYLBXUP8CIVopEmhg0WMKhVeNY?usp=sharing)

Run the following **scripts** for training and/or testing

```python
python train.py # For training the model [--mode=finetune/transfer/scratch]
python test.py test # For testing the model on sample images
python eval.py imdb_small/eval # For evaluating the model on new dataset
```

### Training results

|    | Accuracy | Size | Training Time | Training Mode |
|----|----|----|----|-----|
| **VGG11** | 96.73 | 515.3 MB  |  900 mins |  scratch |
| **Resnet18**  | 99.85  | 44.8 MB |  42 mins |  finetune |
| **MobilenetV2**  | 97.72  | 9.2 MB | 32 mins | transfer |

**Batch size**: 64, **GPU**: Tesla K80

Both **Resnet18 and MobilenetV2**(transfer leraning) were trained for **10 epochs**; whereas **VGG11**(training from scratch) was trained for **100 epochs**.


### Training graphs

**Resnet18:-** 

Finetuning the pretrained resnet18 model.
![Screenshot](results/resnet18.png)

**Mobilenetv2:-**

Mobilenetv2 as a fixed feature extractor.
![Screenshot](results/mobilenetv2.png)

### Sample outputs

Sample classification results

![Screenshot](results/output.png)

### Observations


### Versioning

Version 1.0

### Authors

Anil Sathyan

### Acknowledgments
* "https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html"
* "https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html"
* "https://www.learnopencv.com/image-classification-using-transfer-learning-in-pytorch/"
* "https://towardsdatascience.com/https-medium-com-dinber19-take-a-deeper-look-at-your-pytorch-model-with-the-new-tensorboard-built-in-513969cf6a72"
* "https://www.aiworkbox.com/lessons/how-to-define-a-convolutional-layer-in-pytorch#lesson-transcript-section"
* "https://medium.com/udacity-pytorch-challengers/ideas-on-how-to-fine-tune-a-pre-trained-model-in-pytorch-184c47185a20"
* "https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict"
* "https://www.kaggle.com/c/understanding_cloud_organization/discussion/112582"
* "https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial"
