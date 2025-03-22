import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Neural Network
class CNN(nn.Module):
  def __init__(self, in_channel=3, num_classes=10):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    self.fc1 = nn.Linear(in_features=16*7*7, out_features=num_classes)
  
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.fc1(x)

    return x

# Hyperparameters
input_size = 28*28
in_channel = 3
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load data
train_dataset = datasets.CIFAR10(root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

def plot_cifar():
  # CIFAR-10 class names
  classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  # Create a figure with 2×5 subplots
  fig, axes = plt.subplots(2, 5, figsize=(15, 6))
  axes = axes.flatten()

  # Dictionary to track if we've found an example of each class
  examples_found = {i: False for i in range(10)}


  # Function to convert tensor to displayable image
  def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    # Undo normalization if applied
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


  # Iterate through dataset until we find one example of each class
  for i, data in enumerate(train_loader, 0):
    images, labels = data

    # Check each image in the batch
    for j in range(len(labels)):
      label = labels[j].item()
      if not examples_found[label]:
        # Display this example
        axes[label].imshow(imshow(images[j]))
        axes[label].set_title(classes[label])
        axes[label].axis('off')
        examples_found[label] = True

    # Break if we've found all classes
    if all(examples_found.values()):
      break

  plt.tight_layout()
  plt.show()
plot_cifar()
exit(1)

# Initialize Network on Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = CNN(in_channel=3, num_classes=num_classes).to(device=device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
for epoch in range(num_epochs):
  for batch_idx, (x, y) in enumerate(train_loader):
    # get data on device
    x = x.to(device=device)
    y = y.to(device=device)
    
    # flattern channel * width * height
    x = x.view(x.shape[0], -1)
    
    # forward
    scores = model(x)
    loss = criterion(scores, y)
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    
    # gradient descent or adam step
    optimizer.step()
    
# Check model accuracy on training & test
def check_accuracy(loader, model):
  if loader.dataset.train:
    print('Checking accuracy on train data.')
  else:
    print('Checking accuracy on test data.')
    
  num_correct, num_samples = 0, 0
  model.eval()
  
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device=device)
      y = y.to(device=device)
      x = x.view(x.shape[0], -1)
      
      scores = model(x)
      _, predictions = scores.max(1)
      num_correct += (predictions == y).sum()    
      num_samples += predictions.size(0)
    
    acc = float(num_correct) / float(num_samples)
    print(f'accuracy is {acc * 100:.2f}%')
  
  model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)