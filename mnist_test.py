import torch as t
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt

def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    out_dir = './datasets'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

mnist = mnist_data()