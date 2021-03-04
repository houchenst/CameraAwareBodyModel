import random

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

    def contains(self, point):
        '''
        Returns 1 if a point [x,y,z] falls within the cuboid, 0 otherwise
        '''
        x_in = abs(self.center[0] - point[0]) < self.width/2.
        y_in = abs(self.center[1] - point[1]) < self.length/2.
        z_in = abs(self.center[2] - point[2]) < self.height/2.

        return float(x_in and y_in and z_in)

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
