import numpy as np
import torch
from PIL import Image
from utility import model_device, load_dataloaders
from torchvision import models
from torch import nn, optim


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image, mode="r")
    image.thumbnail((256, 256)) # Resize imge

    # Crop center of image
    width, height = image.size   # Get dimensions
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))

    np_image = np.array(image)
    np_image = np_image / 255 # Normalize color channels

    # Transpose image dimensions
    np_image = np.transpose(np_image, (2, 0, 1))

    # Normalize image
    means = np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
    np_image = (np_image - means) / std
    return np_image


def prediction(image_path, model_checkpoint, k=5):
  # Assign device
  device = model_device()

  model = load_model(model_checkpoint)
  model.to(device)
  model.eval()
  with torch.no_grad():
    image = (torch.from_numpy(process_image(image_path)))
    image = image.type(torch.FloatTensor)
    image = image.to(device)

    log_ps = model(image.reshape((1,3,224,224)))
    ps = torch.exp(log_ps)
    top_ps, top_classes = ps.topk(k, dim=1)
  return (top_ps, top_classes)


def load_model(checkpoint):
  """
  Loads model from `checkpoint.pth` file
  """
  # Load vgg model
  model = models.vgg19(pretrained=True)

  # Freeze network parameters
  for param in model.parameters():
    param.requires_grad = False

  # Create classifier with checkpoint
  checkpoint = torch.load(checkpoint)
  classifier = nn.Sequential(nn.Linear(in_features=checkpoint['input_units'], out_features=checkpoint['hidden_units'][0]),
                            nn.ReLU(),
                            nn.Dropout(p=checkpoint['dropout']),
                            nn.Linear(in_features=checkpoint['hidden_units'][0], out_features=checkpoint['hidden_units'][1]),
                            nn.ReLU(),
                            nn.Dropout(p=checkpoint['dropout']),
                            nn.Linear(in_features=checkpoint['hidden_units'][1], out_features=checkpoint['hidden_units'][2]),
                            nn.ReLU(),
                            nn.Dropout(p=checkpoint['dropout']),
                            nn.Linear(in_features=checkpoint['hidden_units'][2], out_features=checkpoint['output_units']),
                            nn.LogSoftmax(dim=1))
  model.classifier = classifier
  model.classifier.load_state_dict(checkpoint['classifier_state'])
  return model


def save_model(filename, input_units, output_units, hidden_units, model, epochs=5, optimizer=None, dropout_rate=0.3):
    # Save model
    checkpoint = {'input_units': input_units, 'output_units': output_units,
                'hidden_units': hidden_units, 'classifier_state': model.classifier.state_dict(),
                'epochs': epochs, 'optimizer_state': optimizer.state_dict(),
                'dropout': dropout_rate} # Save model information for prediction and for further trainiing

    # Save model
    torch.save(checkpoint, 'checkpoint.pth')


def train_model():
    # Load GTSRB dataset
    dataloaders = load_dataloaders()

    # Assign training device
    device = model_device()

    model = models.vgg19(pretrained=True)
    # inputs: 25088, outputs: 43

    # Freeze network parameters
    for param in model.parameters():
        param.requires_grad = False

    # Create classifier
    input_units, output_units = 25088, 43
    hidden_units = [1024, 512, 256]
    dropout_rate = 0.3

    classifier = nn.Sequential(nn.Linear(in_features=input_units, out_features=hidden_units[0]),
                                nn.ReLU(),
                                nn.Dropout(p=dropout_rate),
                                nn.Linear(in_features=hidden_units[0], out_features=hidden_units[1]),
                                nn.ReLU(),
                                nn.Dropout(p=dropout_rate),
                                nn.Linear(in_features=hidden_units[1], out_features=hidden_units[2]),
                                nn.ReLU(),
                                nn.Dropout(p=dropout_rate),
                                nn.Linear(in_features=hidden_units[2], out_features=output_units),
                                nn.LogSoftmax(dim=1))

    # Replace model's classifier
    model.classifier = classifier
    model.to(device)

    # Define training parameters
    learning_rate = 0.001
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    epochs = 5
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
        with torch.no_grad():
            for images, labels in dataloaders[1]:
                # Move images and labels to GPU
                images, labels = images.to(device), labels.to(device)

                # Make prediction
                log_prediction = model.forward(images)
                prediction = torch.exp(log_prediction)

                # Compute validation loss
                loss = criterion(log_prediction, labels)
                validation_loss += loss.item()

                # Find predicted_labels
                top_ps, top_classes = prediction.topk(k=1, dim=1)

                # Compute accuracy
                equality = top_classes.view(*labels.shape) == labels
                accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
            print(f"Epoch: {i+1}")
            print(f"Training Loss: {training_loss / len(dataloaders[0]):.2f}")
            print(f"Validation Loss: {validation_loss / len(dataloaders[1]):.2f}")
            print(f"Accuracy: {(accuracy / len(dataloaders[1])*100):.2f}%")
            print()
        model.train()

    # Save model
    save_model(filename='checkpoint.pth', input_units=input_units, output_units=output_units, hidden_units=hidden_units,
               model=model, optimizer=optimizer, epochs=epochs, dropout_rate=dropout_rate)
    return model