import torch 
from torchvision import transforms, datasets

def model_device():
    return 'cuda' if torch.cuda.is_available() else "cpu"


def load_dataloaders():
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
    dataloaders = [torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True) for dataset in image_datasets]
    return dataloaders

