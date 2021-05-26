'''
Loads mesh objects and implements a 3D occupancy function for the mesh
Not functional yet
'''

from tk3dv.nocstools.obj_loader import Loader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

OBJ_FILE = "C:\\Users\Trevor\\Brown\\ivl-research\\data\\TriMesh.obj"

mesh = Loader(OBJ_FILE)

# print(mesh.faces)
# print(mesh.vertices)
# mesh.draw()

def cross2d(a,b):
    '''
    2d cross product of a and b
    '''
    return a[0]*b[1] - a[1]*b[0]

def cross3d(a,b):
    '''
    3d cross product of a and b
    '''
    cx = a[1]*b[2] - a[2]*b[1]
    cy = a[2]*b[0] - a[0]*b[2]
    cz = a[0]*b[1] - a[1]*b[0]
    return [cx,cy,cz]

def occupied_3d(vertices, faces, q):
    '''
    True if the mesh represented by the vertices and faces contains the query point, q
    '''
    # if parity of faces directly above q is odd, then q is occupied
    faces_above = 0
    # start=True
    for f in faces:
        p1 = list(vertices[f[0][0]])
        p2 = list(vertices[f[1][0]])
        p3 = list(vertices[f[2][0]])
        
        # check whether this face is entirely above or below q in x
        x_near = (p1[0] > q[0]) + (p2[0] > q[0]) + (p3[0] > q[0])
        if x_near == 0  or x_near == 3:
            continue
        # check whether this face is entirely above or below q in y
        y_near = (p1[1] > q[1]) + (p2[1] > q[1]) + (p3[1] > q[1])
        if y_near == 0 or y_near == 3:
            continue

        # if start:
        #     print(p1)
        #     print(p2)
        #     print(p3)
            
        # check if it is occupied in 2d when projected down z
        e1 = [b-a for a,b in zip(p1,p2)]
        e2 = [b-a for a,b in zip(p2,p3)]
        e3 = [b-a for a,b in zip(p3,p1)]

        # angles should all be clockwise or anticlockwise if it is occupied
        angles = [cross2d([b-a for a,b in zip(q,p)], e) > 0 for e,p in zip([e1,e2,e3],[p1,p2,p3])]
        # if start:
        #     print(angles)
        
        if sum(angles) != 0 and sum(angles) != 3:
            # start=False
            continue
        
        # if the q point is within the bounds of the face in x and y, we will check to 
        # see whether q is above or below the plane in z
        face_normal = cross3d(e1, e2)
        
        # it's below if the dot product is - and the z component of the normal is +
        # or if the dot product is + and the z component of the normal is -
        dot_pos = sum([face_normal[i]*q[i] for i in range(3)]) > 0
        z_pos = face_normal[2] > 0

        faces_above += (dot_pos != z_pos)
        # start=False
    # print(faces_above)
    return bool(faces_above % 2)

# make mesh in bbox 0->1
zs = [v[2] for v in mesh.vertices]
ys = [v[1] for v in mesh.vertices]
xs = [v[0] for v in mesh.vertices]
zmax = max(zs)
zmin = min(zs)
ymax = max(ys)
ymin = min(ys)
xmax = max(xs)
xmin = min(xs)
mesh.vertices = [((x-xmin)/(xmax-xmin), (y-ymin)/(ymax-ymin), (z-zmin)/(zmax-zmin)) for x,y,z in mesh.vertices]

print(occupied_3d(mesh.vertices, mesh.faces, (1,1,1)))
print(occupied_3d(mesh.vertices, mesh.faces, (0,0,0)))
print(occupied_3d(mesh.vertices, mesh.faces, (-1.4575, 15.1137,0)))
print(occupied_3d(mesh.vertices, mesh.faces, (0.5,0.5,0.5)))
# print(mesh.vertices[mesh.faces[0][0]])

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')

samps = np.linspace(0,1,4)

xs,ys,zs = np.meshgrid(samps, samps, samps)
xs = xs.flatten()
ys = ys.flatten()
zs = zs.flatten()

plot_xs = []
plot_ys = []
plot_zs = []

for i in range(xs.shape[0]):
    print(i)
    if occupied_3d(mesh.vertices, mesh.faces, (xs[i], ys[i], zs[i])):
        plot_xs.append(xs[i])
        plot_ys.append(ys[i])
        plot_zs.append(zs[i])

ax1.scatter(xs,ys,zs)
plt.show()

