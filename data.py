import torch
from torch.utils.data import Dataset
import random
import numpy as np
from utils import rotation_matrix, random_cam_center
from tqdm import tqdm

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


class OccDataset2D(Dataset):

    def __init__(self, obj, num_samples, focal_length, sensor_bounds, sensor_size, cam_radius, ray_samples=20, min_cameras=30):
        '''
        :param obj: An object that has an occupancy function and a surface sampling function
        :param num_samples: The number of individual 2D occupancy points to generate for training
        :param sensor_bounds: The bounds of the sensor in 2d camera coordinates. [a,b] will select the 
                              box ranging from -a to a in x and -b to b in y
        :param sensor_size: The resolution of 2d occupancy images that we generate (wxh in pixels)
        :param focal_length: The focal length to use for the camera when making 2D occupancy images
        :param cam_radius: The radius of the sphere from which cam centers should be sampled
        :param ray_samples: The number of 3D occupancy queries to make along a ray to see if the 2D point is occupied
        :param min_cameras: The minimum number of camera locations to include data for. 
                            Pass 1 to force using every pixel for each camera
        '''
        # store inputs
        self.obj = obj
        self.num_samples = num_samples
        self.focal_length = focal_length
        self.sensor_bounds = sensor_bounds
        self.sensor_size = sensor_size
        self.cam_radius = cam_radius
        self.ray_samples = ray_samples

        self.data = []
        self.cam_params = []
        self.labels = []

        # we want a variety of camera views, so if the sensor size is large, we might not use every pixel
        self.samples_per_camera = int(min(self.sensor_size[0]*self.sensor_size[1], self.num_samples / min_cameras))
        
        print("Generating 2D Data...")
        pbar = tqdm(total=num_samples)
        while len(self.data) < self.num_samples:
            cam_center = random_cam_center(self.cam_radius)
            l, x, y, _, _ = obj.contains_2d(cam_center, self.focal_length, self.sensor_bounds, self.sensor_size, ray_samples=self.ray_samples)
            l = l.flatten()
            x = x.flatten()
            y = y.flatten()
            r = rotation_matrix(cam_center, inverse=True).flatten()
            cam_params = list(r) + cam_center

            # samples some fraction of the pixels to add to our data
            for i in random.sample(range(l.shape[0]), self.samples_per_camera):
                self.labels.append(torch.tensor(l[i], dtype=torch.float32))
                self.data.append(torch.tensor([x[i],y[i]], dtype=torch.float32))
                self.cam_params.append(torch.tensor(cam_params, dtype=torch.float32))
            pbar.update(self.samples_per_camera)
        tqdm._instances.clear()

        self.length = len(self.data)


    def __len__(self):
        '''
        Returns the length of the dataset
        '''
        return self.length

    def __getitem__(self, idx):
        '''
        Gets a tuple of (point, label) at given index
        '''
        return self.data[idx], self.labels[idx], self.cam_params[idx]

class OccDataset3DUVD(Dataset):

    def __init__(self, obj, num_samples, focal_length, sensor_bounds, sensor_size, cam_radius, ray_samples=20, min_cameras=30):
        '''
        :param obj: An object that has an occupancy function and a surface sampling function
        :param num_samples: The number of individual 2D occupancy points to generate for training
        :param sensor_bounds: The bounds of the sensor in 2d camera coordinates. [a,b] will select the 
                              box ranging from -a to a in x and -b to b in y
        :param sensor_size: The resolution of 2d occupancy images that we generate (wxh in pixels)
        :param focal_length: The focal length to use for the camera when making 2D occupancy images
        :param cam_radius: The radius of the sphere from which cam centers should be sampled
        :param ray_samples: The number of 3D occupancy queries to make along a ray to see if the 2D point is occupied
        :param min_cameras: The minimum number of camera locations to include data for. 
                            Pass 1 to force using every pixel for each camera
        '''
        # store inputs
        self.obj = obj
        self.num_samples = num_samples
        self.focal_length = focal_length
        self.sensor_bounds = sensor_bounds
        self.sensor_size = sensor_size
        self.cam_radius = cam_radius
        self.ray_samples = ray_samples

        self.uv = []
        self.xyz = []
        self.cam_params = []
        self.labels = []

        # we want a variety of camera views, so if the sensor size is large, we might not use every pixel
        self.samples_per_camera = int(min(self.sensor_size[0]*self.sensor_size[1], self.num_samples / min_cameras))
        
        print("Generating 3D UVD Data...")
        pbar = tqdm(total=num_samples)
        while len(self.xyz) < self.num_samples:
            cam_center = random_cam_center(self.cam_radius)
            _, u, v, occ_3d, coords_3d = obj.contains_2d(cam_center, self.focal_length, self.sensor_bounds, self.sensor_size, ray_samples=self.ray_samples)
            coords_3d = np.reshape(coords_3d, (-1,3))
            occ_3d = occ_3d.flatten()
            u = np.stack([u]*5, axis=-1).flatten()
            v = np.stack([v]*5, axis=-1).flatten()
            r = rotation_matrix(cam_center, inverse=True).flatten()
            cam_params = list(r) + cam_center

            # samples some fraction of the pixels to add to our data
            for i in random.sample(range(coords_3d.shape[0]), self.samples_per_camera):
                self.labels.append(torch.tensor(occ_3d[i], dtype=torch.float32))
                self.xyz.append(torch.tensor(coords_3d[i], dtype=torch.float32))
                self.uv.append(torch.tensor([u[i], v[i]], dtype=torch.float32))
                self.cam_params.append(torch.tensor(cam_params, dtype=torch.float32))
            pbar.update(self.samples_per_camera)
        tqdm._instances.clear()

        self.length = len(self.xyz)


    def __len__(self):
        '''
        Returns the length of the dataset
        '''
        return self.length

    def __getitem__(self, idx):
        '''
        Gets a tuple of (point, label) at given index
        '''
        return self.uv[idx], self.xyz[idx], self.labels[idx], self.cam_params[idx]