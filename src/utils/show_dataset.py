import torchvision
from torchvision import datasets, models, transforms
from matplotlib import pyplot as plt
import random


def simple_dataset(images_path, annotation_json):
    dataset = datasets.CocoCaptions(
        root=images_path,
        annFile=annotation_json,
        transform=transforms.ToTensor())
    print('\nNumber of samples: ', len(dataset))
    return dataset


def show_sample(dataset, sample_idx):
    img, target = dataset[sample_idx]
    print("\nReferences:")
    for label in target:
        print(label)
    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()


def dataset_sample_example(images_path, annotation_json, seed=0):
    dset = simple_dataset(images_path, annotation_json)
    random.seed(seed)
    sample_id = random.randint(0, len(dset))
    show_sample(dset, sample_id)
