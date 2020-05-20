import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#function to filter trainset data
def filter(data, classes):
    batch = []
    val = []
    i = 0
    for images, labels in data:
        if labels in classes:
            batch.append([images, labels])
    return batch

#function to check batch's classes
def checkBatchClasses(batch):
    setlab = set()
    for images, labels in batch1:
        # print(labels)
        if labels not in setlab:
            setlab.add(labels)
    print(setlab)
    print(len(setlab))

def getTrainVal(batch):
    curr = batch
    random.shuffle(curr)
    limit = int((len(curr) * 2) / 3)
    train = range(0, limit)
    val = range(limit, len(curr))
    return train, val

def retrieveIndexTrainVal(batch):
    samples = range(0,len(batch))
    labels = []
    for i in samples:
        labels.append(batch[i][1])
    train, val, y_train, y_val = train_test_split(samples,labels,test_size=0.1,
    random_state=42,stratify=labels)
    index_train = []
    index_val = []
    for i, el in enumerate(samples):
        if el in train:
            index_train.append(i)
        else:
            index_val.append(i)
    print("retrieveTrainVal")
    print(len(index_train))
    print(len(index_val))
    return index_train, index_val

def retrieveDataTrainVal(batch, ind_train, ind_val):
    train_dataset = Subset(batch, ind_train)
    val_dataset = Subset(batch, ind_val)
    return train_dataset, val_dataset

batch1 = filter(trainset, list(range(10, 20)))
print("Batch1 size: {}".format(len(batch1)))
checkBatchClasses(batch1)

ind_train, ind_val = retrieveIndexTrainVal(batch1)
print("ind_train size: {}".format(len(ind_train)))
print("ind_val size: {}".format(len(ind_val)))

batch_train_data, batch_val_data = retrieveDataTrainVal(batch1, ind_train, ind_val)


batch1_loader = torch.utils.data.DataLoader(batch1, batch_size=4,
                                          shuffle=True, num_workers=0)

train_batch1_loader = torch.utils.data.DataLoader(batch_train_data, batch_size=4,
                                          shuffle=True, num_workers=0)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

# get some random training images
dataiter = iter(batch1_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))

dataiter = iter(train_batch1_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
#
# images, labels = train_batch1_loader.dataset.__getitem__(80)
# imshow(torchvision.utils.make_grid(images))

images, labels = batch1_loader.dataset.__getitem__(5)
imshow(torchvision.utils.make_grid(images))



