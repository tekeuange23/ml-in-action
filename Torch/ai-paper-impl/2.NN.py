import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Neural Network
class NN(nn.Module):
  def __init__(self, input_size, num_classes):
    super(NN, self).__init__()
    self.fc1 = nn.Linear(in_features=input_size, out_features=50)
    self.fc2 = nn.Linear(in_features=50, out_features=num_classes)
  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# Hyperparameters
input_size = 28*28
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load data
train_dataset = datasets.MNIST(root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Visualize an image Create a 2*5 subplot figure
fig, axs = plt.subplots(2, 5, figsize=(15, 6))

# Loop through each class (0-9) and plot an image
for i in range(10):
    images = []
    labels = []
    
    # Get all images from the current class
    for j, (x, y) in enumerate(train_dataset):
        if y == i:
            images.append(x)
            labels.append(y)
            
            # Add a maximum of 5 images to avoid overfilling the figure
            if len(images) >= 5:
                break
                
    # Plot the images horizontally on the same subplot
    for j, img in enumerate(images):
        axs[i // 5, i % 5].imshow(img.squeeze(), cmap='gray')
        
    # Label each subplot
    axs[i // 5, 0].set_title(f'Class {i}')

# Layout so plots do not overlap
fig.tight_layout()

plt.show()


# Initialize Network on Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = NN(input_size=input_size, num_classes=num_classes).to(device=device)

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