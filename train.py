from torchvision import datasets, transforms, models
from torch import nn, optim, utils, cuda

# Define dataset transforms
train_transform = transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.RandomRotation(30),
                                     transforms.RandomHorizontalFlip(p=0.4), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Load datasets
image_datasets = [datasets.GTSRB(root='GTSRB/train', download=True, split='train', transform=train_transform),
                  datasets.GTSRB(root='GTSRB/test', download=True, split='test', transform=test_transform)]


# Download vgg19 model for transfer learning
model = models.vgg19(pretrained=True) # inputs: 25088, outputs: 43

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

# Define training parameters
learning_rate = 0.001
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


