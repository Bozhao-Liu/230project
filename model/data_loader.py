import random
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import pdb
import torch
import torch.utils.data
import torchvision

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
def train_transformer_list(params):
    train_transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=params.num_input_channels), # convert RGB image to greyscale (optional, 1 vs. 3 channels)
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.RandomVerticalFlip(),  # randomly flip image vertically
        transforms.RandomRotation(180), # randomly rotate image by 180 degrees
        transforms.ToTensor()])  # transform it into a torch tensor
    return train_transformer

# loader for evaluation, no horizontal flip
def eval_transformer_list(params):
    eval_transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=params.num_input_channels), 
        # transforms.Resize([177, 128]),  # resize the image to 177x128 (remove if images are already 64x64)
        transforms.ToTensor()])  # transform it into a torch tensor
    return eval_transformer


class FundusDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg') or f.endswith('.png')]

        # self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
        self.labels = []
        for filename in self.filenames:
            imagename = os.path.split(filename)[-1]
            ind = imagename.find('(')
            label = imagename[(ind+1):(ind+3)]
            finallabel = np.asarray([int(i) for i in label])
            self.labels.append(finallabel)
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)

        return image, self.labels[idx]


# adopted from https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/sampler.py
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        self.labels = dataset.labels

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = str(self.labels[idx])
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[str(self.labels[idx])] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples 


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_data".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                train_dataset = FundusDataset(path, train_transformer_list(params))
                dl = DataLoader(train_dataset,                     
                                    sampler=ImbalancedDatasetSampler(train_dataset),
                                    batch_size=params.batch_size, shuffle=False, # sampler option is mutually exclusive with shuffle
                                    num_workers=params.num_workers,
                                    pin_memory=params.cuda)
            else:
                eval_dataset = FundusDataset(path, eval_transformer_list(params))
                dl = DataLoader(eval_dataset, 
                                    batch_size=params.batch_size, shuffle=False,
                                    num_workers=params.num_workers,
                                    pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
