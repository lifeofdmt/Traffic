from torchvision import datasets, transforms, models
from torch import nn, optim, utils, cuda, no_grad, exp, FloatTensor, mean

# Define dataset transforms
train_transform = transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(p=0.4), transforms.ToTensor(), 
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Load datasets
image_datasets = [datasets.GTSRB(root='GTSRB/train', download=True, split='train', transform=train_transform),
                  datasets.GTSRB(root='GTSRB/test', download=True, split='test', transform=test_transform)]

# Create dataloaders
dataloaders = [utils.data.DataLoader(dataset, batch_size=64, shuffle=True) for dataset in image_datasets]

device = 'cuda' if cuda.is_available() else "cpu"
print(device)

model = models.vgg19(pretrained=True)
# inputs: 25088, outputs: 43

# Freeze network parameters
for param in model.parameters():
  param.requires_grad = False

# Create classifier
dropout_rate = 0.3

classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=1024),
                            nn.ReLU(),
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(in_features=1024, out_features=512),
                            nn.ReLU(),
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(in_features=512, out_features=256),
                            nn.ReLU(),
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(in_features=256, out_features=43),
                            nn.LogSoftmax(dim=1))

# Replace model's classifier
model.classifier = classifier
model.to(device)

# Define training parameters
learning_rate = 0.001
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
epochs = 10
criterion = nn.NLLLoss()

# Train network
for i in range(epochs):
  training_loss = 0
  for images, labels in dataloaders[0]:
    # Move images and labels to GPU
    images, labels = images.to(device), labels.to(device)

    # Clear gradients
    optimizer.zero_grad()

    # Compute Loss
    log_prediction = model.forward(images)
    loss = criterion(log_prediction, labels)
    loss.backward()

    optimizer.step() # Update model parameters

    # Add up training_loss
    training_loss += loss.item()
  
  # Check models accuracy on Validation Set
  validation_loss = 0
  accuracy = 0
  model.eval()
  
  with no_grad():
    for images, labels in dataloaders[1]:
      # Move images and labels to GPU
      images, labels = images.to(device), labels.to(device)

      # Make prediction
      log_prediction = model.forward(images)
      prediction = exp(log_prediction)

      # Compute validation loss
      loss = criterion(log_prediction, labels)
      validation_loss += loss.item()

      # Find predicted_labels
      top_ps, top_classes = prediction.topk(k=1, dim=1) 

      # Compute accuracy
      equality = top_classes.view(*labels.shape) == labels
      accuracy += mean(equality.type(FloatTensor)).item()
    
  print(f"Epoch: {i+1}")
  print(f"Training Loss: {training_loss / len(dataloaders[0]):.2f}")
  print(f"Validation Loss: {validation_loss / len(dataloaders[1]):.2f}")
  print(f"Accuracy: {(accuracy / len(dataloaders[1])*100):.2f}%")
  print()
  model.train()

