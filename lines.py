import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from utils import *

def plot_nearest_point(line, ax, point=[0,0,0]):
    '''
    Plots the point on a line closest to another point
    line should be [x0,y0,z0,a,b,c] such that [x,y,z] = [x0,y0,z0] + t[a,b,c]
    '''
    p0 = np.array(line[:3])
    p1 = np.array([line[0]+line[3], line[1]+line[4], line[2]+line[5]])
    q = np.array(point)
    # solve for t from first derivative of squared distance wrt t set to 0
    # t = -1 * ((p1-p0)*(p0-q)) / (np.square(p1-p0))

    # solve for when dot product of po->pr and p1->pr is 0 using quadratic formula
    a = np.sum(np.square(line[3:]))
    b = np.sum(np.array(line[3:]) * (p0-q))
    # c = 0 so we get sqrt(b**2)
    rad_term = abs(b)
    t1 = (-b + rad_term) / (2*a)
    t2 = (-b - rad_term) / (2*a)

    print(np.sum((t1*np.array(line[3:])) * (np.array(line[:3]) + t1*np.array(line[3:]) - np.array(point))))
    print(np.sum((t2*np.array(line[3:])) * (np.array(line[:3]) + t2*np.array(line[3:]) - np.array(point))))

    
    xs = np.array([line[0]]*3) + np.array([-2,0,2])*line[3]
    ys = np.array([line[1]]*3) + np.array([-2,0,2])*line[4]
    zs = np.array([line[2]]*3) + np.array([-2,0,2])*line[5]
    ax.plot(xs,ys,zs, color="tab:blue")
    
    xs = np.array([line[0]]*2) + np.array([t1,t2])*line[3]
    ys = np.array([line[1]]*2) + np.array([t1,t2])*line[4]
    zs = np.array([line[2]]*2) + np.array([t1,t2])*line[5]
    ax.scatter(xs,ys,zs, color="tab:orange")

    # show q
    ax.scatter(point[0], point[1], point[2], color="tab:red")
    # show p0
    ax.scatter(line[0], line[1], line[2], color="tab:red")


def reparameterize_line(p0, p1, q=[0,0,0]):
    '''
    Takes in a line parameterized as two points, p0, p1 
    Outputs as line parameterized as follows
    v - the point on the line closest to the point q
    theta - a measure of angle in the plane orthogonal to the q-center vector
    '''
    p0 = np.array(p0)
    p1 = np.array(p1)
    q = np.array(q)
    direction = p1-p0

    # solve for when dot product of po->p1 and p1->q is 0 using quadratic formula
    a = np.sum(direction)
    b = np.sum(direction * (p0-q))
    # c = 0 so we get sqrt(b**2)
    rad_term = abs(b)
    t1 = (-b + rad_term) / (2*a)
    t2 = (-b - rad_term) / (2*a)

    sol1 = p0 + t1*direction
    sol2 = p0 + t2*direction

    v = sol1
    if np.sum(np.square(sol2-q)) < np.sum(np.square(sol1-q)):
        v=sol2
    
    base = angle_reference_vector(v)
    unit_dir = direction / math.sqrt(np.sum(np.square(direction)))
    # base and unit_dir are unit vectors, so their dot product is the cosine of their angle
    theta = math.acos(np.sum(base*unit_dir))

    return v, theta

    

def angle_reference_vector(v, center=[0,0,0]):
    '''
    Outputs a vector in the plane orthogonal to the v-center vector that
    is consistent across calls.
    '''
    v1 = v[0]-center[0]
    v2 = v[1]-center[1]
    v3 = v[2]-center[2]

    # let x and y of our new vector be 1. Solve for z and normalize
    # a and b are orthogonal iff a dot b = 0
    # we know a, are choosing to let the first two elements of b be 1,
    #  and are  solving for the third
    if v3 == 0:
        return [0,0,1]
    else:
        z = -(v1 + v2)/v3
        normalizer = (1 + 1 + z**2)**0.5
        return [1/normalizer, 1/normalizer, z/normalizer]

def plot_reference_vectors(num_vectors=100):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for _ in range(num_vectors):
        v = random_cam_center(4.)
        base = angle_reference_vector(v)
        xs = [v[0], v[0]+base[0]]
        ys = [v[1], v[1]+base[1]]
        zs = [v[2], v[2]+base[2]]
        ax.plot(xs,ys,zs, color="tab:blue")
        ax.scatter(v[0], v[1], v[2], color="tab:red")
    plt.show()


if __name__ == "__main__":
    plot_reference_vectors()
    # print(reparameterize_line([1,1,1], [1,1, 2]))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plot_nearest_point([1,1,1,1,.5,.5], ax)
    # plot_nearest_point([1,1,1,1,0,0], ax)
    # plt.show()

