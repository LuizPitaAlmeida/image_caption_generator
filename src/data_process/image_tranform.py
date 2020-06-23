from torchvision import transforms

IMG_SIZE = 224
IMG_MEAN_VALUES = (0.485, 0.456, 0.406)
IMG_STD_VALUES = (0.229, 0.224, 0.225)


def transform_function():
    transform = transforms.Compose([
        #transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN_VALUES,
                             IMG_STD_VALUES)])
    return transform
