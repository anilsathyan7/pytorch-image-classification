import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import multiprocessing
from sklearn.metrics import confusion_matrix

# Paths for image directory and model
EVAL_DIR=sys.argv[1])
EVAL_MODEL='models/mobilenetv2.pth'

# Load the model for evaluation
model = torch.load(EVAL_MODEL)
model.eval()

# Configure batch size and nuber of cpu's
num_cpu = multiprocessing.cpu_count()
bs = 8

# Prepare the eval data loader
eval_transform=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

eval_dataset=datasets.ImageFolder(root=EVAL_DIR, transform=eval_transform)
eval_loader=data.DataLoader(eval_dataset, batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True)

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of classes and dataset-size
num_classes=len(eval_dataset.classes)
dsize=len(eval_dataset)

# Class label names
class_names=['apple','atm card','cat','banana','bangle','battery','bottle','broom','bulb','calender','camera']

# Initialize the prediction and label lists
predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

# Evaluate the model accuracy on the dataset
correct = 0
total = 0
with torch.no_grad():
    for images, labels in eval_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        predlist=torch.cat([predlist,predicted.view(-1).cpu()])
        lbllist=torch.cat([lbllist,labels.view(-1).cpu()])

# Overall accuracy
overall_accuracy=100 * correct / total
print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize, 
    overall_accuracy))

# Confusion matrix
conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
print('Confusion Matrix')
print('-'*16)
print(conf_mat,'\n')

# Per-class accuracy
class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
print('Per class accuracy')
print('-'*18)
for label,accuracy in zip(eval_dataset.classes, class_accuracy):
     class_name=class_names[int(label)]
     print('Accuracy of class %8s : %0.2f %%'%(class_name, accuracy))

'''
Sample run: python eval.py eval_ds
'''
