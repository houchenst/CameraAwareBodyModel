import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotation_matrix(cam_center, points_at=(0,0,0), inverse=False):
    '''
    Gets the rotation matrix that rotates world coordinates into camera coordinates.
    cam_center - The center of the camera [x,y,z]
    points_at  - A point that the camera is facing 
                 NOTE: currently only supports cameras facing the origin
    inverse    - If inverse is True, returns the rotation matrix that transforms
                 camera coordinates into world coordinates.
    '''
    cx,cy,cz = cam_center

    # TODO: Figure this out
    # probably just need to take the complementary or negative angle somewhere
    cy = -cy

    # make y rotation matrix
    Nxz = (cx**2 + cz**2)**(0.5)
    if Nxz != 0:
        Ry = np.zeros((3,3))
        Ry[0,0] = -cz/Nxz
        Ry[0,2] = cx/Nxz * (-1 if inverse else 1.)
        Ry[1,1] = 1.
        Ry[2,0] = -cx/Nxz * (-1 if inverse else 1.)
        Ry[2,2] = -cz/Nxz
    else:
        Ry = np.eye(3)

    # make x rotation matrix
    Nxyz = (cx**2 + cy**2 + cz**2)**(0.5)
    if Nxyz != 0:
        Rx = np.zeros((3,3))
        Rx[0,0] = 1.
        Rx[1,1] = Nxz/Nxyz
        Rx[1,2] = -cy/Nxyz * (-1 if inverse else 1.)
        Rx[2,1] = cy/Nxyz * (-1 if inverse else 1.)
        Rx[2,2] = Nxz/Nxyz
    else:
        Rx = np.eye(3)

    R = np.matmul(Ry, Rx)

    return R

def camera_projection(center, focal_length=0.5):
    # make intrinsic matrix
    K = np.zeros((3,3))
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[2,2] = 1.

    # make center offset
    C = np.array(center)

    # make rotation matrix
    R = rotation_matrix(center)

    # define the projection function
    cam_z = lambda x: np.matmul(R, x-C)[2]
    proj = lambda x: np.matmul(K, np.matmul(R, x-C))/cam_z(x)

    return proj


class Cuboid(object):

    def __init__(self, center, width, length, height):
        '''
        A class to represent cuboids
        center - a tuple defining the cuboid center (x,y,z)
        width - cuboid width (x)
        length - cuboid length (y)
        height - cuboid height (z)
        '''

        self.center = center
        self.width = width
        self.length = length
        self.height = height

        self.bounds = ((center[0]-width/2., center[0]+width/2.),
                        (center[1]-length/2., center[1]+length/2.),
                        (center[2]-height/2., center[2]+height/2.))

        # the furthest occupied radius from the origin
        self.max_radius = np.max(np.sqrt(np.sum(np.square(np.reshape(np.array(self.bounds), (-1,3))), axis=1)))
        
        

    def contains(self, point):
        '''
        Returns 1 if a point [x,y,z] falls within the cuboid, 0 otherwise
        '''
        x_in = abs(self.center[0] - point[0]) < self.width/2.
        y_in = abs(self.center[1] - point[1]) < self.length/2.
        z_in = abs(self.center[2] - point[2]) < self.height/2.

        return float(x_in and y_in and z_in)

    def contains_2d(self, center, focal_length, sensor_bounds, sensor_size, ray_samples=100, plot_samples=False):
        '''
        Returns a binary image mask (as numpy array) that shows the 2D occupancy from a camera 
        with focal_length, positioned at center, and looking directly at the origin.
        center        - [x,y,z] defining the camera center
        focal_length  - the focal length of the camera
        sensor_bounds - The bounds of the sensor in 2d camera coordinates. [a,b] will select the 
                        box ranging from -a to a in x and -b to b in y
        sensor_size   - essentially the pixel size of the sensor as [x_pixels, y_pixels]. This 
                        defines the number of 2d samples that will be taken.
        ray_samples   - each 2d point defines a line in 3d space. ray_samples defines the number 
                        of points to sample along this 3d line.
        '''
        # TODO: display the rays in camera and real world coordinates.
        cam_radius = np.sqrt(np.sum(np.square(np.array(center))))
        obj_radius = self.max_radius

        # only supports views that are entirely outside of the occupied space currently
        assert(cam_radius > obj_radius)

        # the ray depths that we will sample
        near = cam_radius-obj_radius
        # the second term accounts for the fact that some rays aren't parallel to z in 3d camera coords
        far = (obj_radius+cam_radius)/(focal_length/np.sqrt(focal_length**2 + max(sensor_bounds)**2))
        # # now in camera coordinates
        # far = far + near
        # near = 0
        cam_space_zs = np.linspace(near, far, num=ray_samples)

        # make x, y and z coordinates in 3d camera space
        x_bound, y_bound = sensor_bounds
        x_pix, y_pix = sensor_size
        xs_2d = np.linspace(-x_bound, x_bound, x_pix)
        ys_2d = np.linspace(-y_bound, y_bound, y_pix)
        xs, ys, zs = np.meshgrid(xs_2d, ys_2d, cam_space_zs)
        
        # the xs and ys are dependent on depth
        xs = xs * (zs/focal_length)
        ys = ys * (zs/focal_length)

        # meshgrid output shape is (num_y, num_x, num_z)
        num_y, num_x, num_z = xs.shape

        
        output = np.zeros((num_y, num_x))
        first=True
        R_inv = rotation_matrix(center, inverse=True)
        # R_inv = np.linalg.inv(rotation_matrix(center))
        C = np.array(center)
        cam2world = lambda x: np.matmul(R_inv, np.array(x)) + C
        for yi in range(num_y):
            for xi in range(num_x):
                output[yi, xi] = max([self.contains(cam2world([xs[yi,xi,zi], ys[yi,xi,zi], zs[yi,xi,zi]])) for zi in range(num_z)])
                # if yi == num_y//2 and xi == num_x//2:
                #     for zi in range(xs.shape[2]):
                #         print(f"Point {cam2world([xs[yi,xi,zi], ys[yi,xi,zi], zs[yi,xi,zi]])} : {self.contains(cam2world([xs[yi,xi,zi], ys[yi,xi,zi], zs[yi,xi,zi]]))}")
                first=False

        # Visualize the sampling rays
        if plot_samples:
            fig = plt.figure()

            # show ray sampling in 3d camera space
            ax1 = fig.add_subplot(121, projection='3d')
            for i in range(num_y):
                for j in range(num_x):
                    ax1.plot(xs[i,j,:], ys[i,j,:], zs[i,j,:], color="tab:blue")
            ax1.scatter([0], [0], [0], color="tab:red")
            

            # show ray sampling in 3d world space
            ax2 = fig.add_subplot(122, projection='3d')
            for xi in range(num_y):
                for yi in range(num_x):
                    x_wrld = [cam2world([xs[yi,xi,zi], ys[yi,xi,zi], zs[yi,xi,zi]])[0] for zi in range(num_z)]
                    y_wrld = [cam2world([xs[yi,xi,zi], ys[yi,xi,zi], zs[yi,xi,zi]])[1] for zi in range(num_z)]
                    z_wrld = [cam2world([xs[yi,xi,zi], ys[yi,xi,zi], zs[yi,xi,zi]])[2] for zi in range(num_z)]
                    # print([cam2world([xs[yi,xi,zi], ys[yi,xi,zi], zs[yi,xi,zi]]) for zi in range(num_z)])
                    ax2.plot(x_wrld, y_wrld, z_wrld, color="tab:blue")
            ax2.scatter([center[0]], [center[1]], [center[2]], color="tab:red")

            plt.show()

        
        return output

    def sample_surface_single(self, gaussian_noise=None):
        '''
        Returns a single point, [x,y,z], sampled from a uniform distribution
        over the surface of the cuboid.
        If gaussian_noise is a float, applies gaussian noise to the sampled point
        with sigma equal to the passed value.
        '''
        ((minx, maxx), (miny, maxy), (minz, maxz)) = self.bounds
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        z = random.uniform(minz, maxz)

        # push one point to boundary so that it's on the edge
        # the probability of picking a specific face of the 
        # cuboid is (area of face)/(total surface area)
        x_face_area = self.length * self.height
        y_face_area = self.width * self.height
        z_face_area = self.width * self.length
        face = random.choices(range(6), weights=2*[x_face_area]+2*[y_face_area]+2*[z_face_area], k=1)

        if face == 0:
            x = minx
        elif face == 1:
            x = maxx
        elif face == 2:
            y = miny
        elif face == 3:
            y = maxy
        elif face == 4:
            z = minz
        elif face == 5:
            z = maxz

        # apply noise if requested
        if gaussian_noise is not None:
            x += random.normalvariate(0, gaussian_noise)
            y += random.normalvariate(0, gaussian_noise)
            z += random.normalvariate(0, gaussian_noise)

        return [x,y,z]

    def sample_surface(self, num_samples, gaussian_noise=None):
        return [self.sample_surface_single(gaussian_noise=gaussian_noise) for _ in range(num_samples)]



if __name__ == "__main__":
    obj = Cuboid((0,0,0), 0.1, 0.3, 0.8)
    bbox = ((-1.,1.), (-1.,1.), (-1.,1.))

    center = (2,2,3)

    # proj_2d = camera_projection(center)

    # points = obj.sample_surface(100000)
    # points_2d = [proj_2d(p) for p in points]

    # ax = plt.subplot(111)

    # xs = [p[0] for p in points_2d]
    # ys = [p[1] for p in points_2d]

    # ax.scatter(xs, ys)

    plt.imshow(obj.contains_2d(center, 2.0, [0.5,0.5], [200,200], ray_samples=40, plot_samples=False))
    obj.contains_2d(center, 2.0, [0.5,0.5], [6,6], ray_samples=10, plot_samples=True)

    plt.show()    
