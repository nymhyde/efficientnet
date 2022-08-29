import os
import random
import time
import numpy as np
import torch
from efficientnet.model import EfficientNet
from torchsummary import summary

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


# plots
import matplotlib.pyplot as plt
pparam = dict(xlabel='Epochs', ylabel='Cross Entropy Loss')


model_name = 'efficientnet-b0'
BATCH_SIZE = 200
SEED = 44
EPOCHS = 2
img_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
])

train_dataset = MNIST(".", train=True, transform=transform, download=True)
test_dataset = MNIST(".", train=False, transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)


model = EfficientNet.from_name(model_name)

image_size = EfficientNet.get_image_size(model_name)        
print(f'\n\nalthough b0 input img size is {image_size}; you can get away with a resize of {img_size}. \n')

# adjust the final linear layer in and out features.
feature = model._fc.in_features
model._fc = torch.nn.Linear(feature,10)

# ---------------------------------------------
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
 '''
# ---------------------------------------------

model.to(device)
child_count = 0
for child in model.children():
    print(f' Child {child_count} is :: ')
    print(child)
    child_count += 1


#summary(model, (1, img_size, img_size))


# ------------------------------------------

torch.manual_seed(SEED)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the model
total_step = len(train_loader)
e_step = len(test_loader)
start_time = time.time()
train_loss = []
val_loss = []
accuracy = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        output = model(images)
        loss = criterion(output, labels)
        epoch_loss += loss.item()

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, EPOCHS, i + 1, total_step, loss.item()))

    epoch_loss = epoch_loss / total_step     
    train_loss.append(epoch_loss)
    print(f'\nTotal loss after epoch {epoch+1} :: {epoch_loss:.4f}')

    # test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        eval_loss = 0.0
        acc = 0.0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            e_loss = criterion(output, labels)
            eval_loss += e_loss.item()

            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
 
        eval_loss = eval_loss / e_step
        val_loss.append(eval_loss)

        acc = (100 * correct / total)
        accuracy.append(acc)
        
        print(f'Test Accuracy of the model on the 10000 test images: {acc:.2f}')
        print(f'Eval Loss of the model on the 10000 test images: {eval_loss:.4f}\n')
        print('*'*25)

train_time = time.time() - start_time
print('*'*25)
print(f'\nTime taken to train the model :: {train_time/60:.2f} minutes.\n')
print('*'*25)


# test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('\nTest Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    print('*'*25)



PATH_1 = 'saved_models/'
PATH_2 = 'figures/'

MODEL = 'mnist.pt'
if not os.path.isdir(PATH_1):
    os.mkdir(PATH_1)
if not os.path.isdir(PATH_2):
    os.mkdir(PATH_2)


torch.save(model, PATH_1+MODEL)



with plt.style.context(['science', 'no-latex', 'grid', 'nature']):
    fig, ax = plt.subplots()
    ax.plot(np.arange(EPOCHS)+1, train_loss, 'royalblue', linewidth=2, marker='o', markersize=3, label='train loss')
    ax.legend(loc='best')
    ax.set_title('Custom Model on MNIST dataset')
    ax.set(**pparam)

    #fig.savefig('figures/fig_train.pdf')
    fig.savefig('figures/fig_train.jpg', dpi=900)



with plt.style.context(['science', 'no-latex', 'grid', 'nature']):
    fig, ax = plt.subplots()
    ax.plot(np.arange(EPOCHS)+1, accuracy, 'orangered', linewidth=2, marker='o', markersize=3, label='Accuracy')
    ax.legend(loc='best')
    ax.set_title('Custom model on MNIST dataset')
    ax.set(**pparam)

    #fig.savefig('figures/fig_test.pdf')
    fig.savefig('figures/fig_test.jpg', dpi=900)



with plt.style.context(['science', 'no-latex', 'grid', 'nature']):
    fig, ax = plt.subplots()
    ax.plot(np.arange(EPOCHS)+1, train_loss, 'royalblue', linewidth=2, marker='o', markersize=3, label='train loss')
    ax.plot(np.arange(EPOCHS)+1, val_loss, 'orangered', linewidth=2, marker='o', markersize=3, label='test loss')
    ax.legend(loc='best')
    ax.set_title('Custom model on MNIST dataset')
    ax.set(**pparam)

    #fig.savefig('figures/fig2.pdf')
    fig.savefig('figures/fig_both_train_test.jpg', dpi=900)



