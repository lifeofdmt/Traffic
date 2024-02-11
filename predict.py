from torch import load, from_numpy
from torchvision import models
from PIL import Image
import numpy as np

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
  model = load_model(model_checkpoint)
  model.to(device)
  model.eval()

  with no_grad():
    image = (from_numpy(process_image(image_path)))
    image = image.type(FloatTensor)
    image = image.to(device)

    log_ps = model(image.reshape((1,3,224,224)))
    ps = exp(log_ps)
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
  checkpoint = load(checkpoint)
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

# Evaluate saved model
model = load_model('checkpoint.pth')
model.to(device)

model.eval()
with no_grad():
    accuracy = 0
    for images, labels in dataloaders[1]:
        images, labels = images.to(device), labels.to(device)

        # Compute testing accuracy
        ps = exp(model(images))
        top_p, top_class = ps.topk(1, dim=1)

        equality = top_class.view(*labels.shape) == labels
        accuracy += mean(equality.type(FloatTensor)).item()
print(f"Testing Accuracy: {(accuracy/len(dataloaders[1]))*100:.2f}%")
print()

top_ps, top_classes = prediction('GTSRB/test/gtsrb/GTSRB/Final_Test/Images/00005.ppm', 'checkpoint.pth')
print(f'Top Probabilities: {top_ps}')
print(f'Top Classes: {top_classes}')