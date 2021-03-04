import torch
from torch.utils.data import Dataset
import random

class OccDataset3D(Dataset):

    def __init__(self, obj, bbox, num_samples, noise=0.03):
        '''
        :param obj: An object that has an occupancy function and a surface sampling function
        :param bbox: A tuple of tuples with the max and min value of each dimension (x,y,z)
        :param num_samples: The number of samples to take from the surface of the object. Twice this
                            number of samples will be taken uniformly from within the bbox
        '''
        # store inputs
        self.obj = obj
        ((self.x_min, self.x_max),(self.y_min, self.y_max),(self.z_min, self.z_max)) = bbox
        self.num_samples = num_samples

        # random uniform sampling within bbox
        self.points = [[random.uniform(self.x_min, self.x_max), random.uniform(self.y_min, self.y_max), random.uniform(self.z_min, self.z_max)] for _ in range(2*num_samples)]
        # surface sampling
        self.points += self.obj.sample_surface(num_samples, gaussian_noise=noise)

        # getting point labels
        self.labels = [torch.tensor(self.obj.contains(x)) for x in self.points]

        # make points tensors
        self.points = [torch.tensor(point) for point in self.points]

        self.length = len(self.points)


    def __len__(self):
        '''
        Returns the length of the dataset
        '''
        return self.length

    def __getitem__(self, idx):
        '''
        Gets a tuple of (point, label) at given index
        '''
        return self.points[idx], self.labels[idx]