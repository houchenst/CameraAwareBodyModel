from tk3dv.nocstools.obj_loader import Loader

OBJ_FILE = "C:\\Users\Trevor\\Brown\\ivl-research\\data\\FinalBaseMesh.obj"

mesh = Loader(OBJ_FILE)

print(mesh.faces)
mesh.draw()