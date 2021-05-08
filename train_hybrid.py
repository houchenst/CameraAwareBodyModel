from comet_ml import Experiment
from torch import nn
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from utils import *
from models import UnstructuredHybrid
from data import OccDataset2D, OccDataset3DUVD

# NASA paper uses batch size 12, learning rate 1e-4
hyperparams = {
    "batch_size" : 12,
    "learning_rate" :  1e-4,
    "epochs" : 20,
    "layers" : 8,
    "hidden_size": 80,

    "focal_length": 2.0,
    "sensor_bounds": [0.5, 0.5],
    "sensor_size": [100, 100],
    "cam_radius": 3.0,
    "3d_ratio": 1,
}
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
device = "cpu"
# print(f"{torch.cuda.get_device_properties(device).total_memory}\n\n")
# print(f"{torch.cuda.memory_reserved(device)}\n\n")
# print(f"{torch.cuda.memory_allocated(device)}\n\n")

def train_2d_epoch(model, train_loader, experiment, epoch, loss_fn, optimizer, step):
    '''
    Trains the model for one epoch on 2d data
    '''
    print(f"2D Training...")
    total_loss = 0
    total_points = 0
    for uv, occ, cam_params in tqdm(train_loader):
        uv = uv.to(device)
        occ = occ.to(device)
        cam_params = cam_params.to(device)
        pred = model(uv, cam_params).reshape((hyperparams["batch_size"],))
        optimizer.zero_grad()
        loss = loss_fn(pred, occ)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_points += hyperparams["batch_size"]
        experiment.log_metric("2D Loss", float(loss.item()), step=step)
        step+=1
    print(f"2D Mean Train Loss: {total_loss/total_points}\n")
    return step

def train_3d_epoch(model, train_loader, experiment, epoch, loss_fn, optimizer, step):
    '''
    Trains the model for one epoch on 3d data
    '''
    print(f"3D Training...")
    total_loss = 0
    total_points = 0
    for uv, xyz, occ, cam_params in tqdm(train_loader):
        uv = uv.to(device)
        xyz = xyz.to(device)
        occ = occ.to(device)
        cam_params = cam_params.to(device)
        pred = model(uv, cam_params, xyz=xyz).reshape((hyperparams["batch_size"],))
        optimizer.zero_grad()
        loss = loss_fn(pred, occ)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_points += hyperparams["batch_size"]
        experiment.log_metric("3D Loss", float(loss.item()), step=step)
        step+=1
    print(f"3D Mean Train Loss: {total_loss/total_points}\n")
    return step


def train(model, train_loader_2d, train_loader_3d, experiment):
    '''
    Trains the model
    '''
    step_2d=0
    step_3d=0
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])
    model = model.train()
    with experiment.train():
        for e in range(hyperparams["epochs"]):
            print(f"\n\n\n----------    EPOCH {e+1}    ----------")
            step_2d = train_2d_epoch(model, train_loader_2d, experiment, e, loss_fn, optimizer, step_2d)
            step_3d = train_3d_epoch(model, train_loader_3d, experiment, e, loss_fn, optimizer, step_3d)
            # log visualizations
            visualize_2d(model, obj, random_cam_center(hyperparams["cam_radius"]), show=False, log=True, experiment=experiment, epoch=e)
            # visualize_3d(model, obj, bbox, show=False, log=True, experiment=experiment, epoch=e)
            torch.save(model.state_dict(), './saved_models/modelhybrid.pt')



def test_2d(model, test_loader, experiment):
    '''
    Tests the model on validation data
    '''
    loss_fn = nn.BCELoss()
    model = model.eval()
    with torch.no_grad(), experiment.test():
        total_loss = 0
        total_points = 0
        total_correct = 0
        for uv,occ, cam_params in tqdm(test_loader):
            uv = uv.to(device)
            occ = occ.to(device)
            cam_params = cam_params.to(device)
            pred = model(uv, cam_params).reshape((hyperparams["batch_size"],))
            loss = loss_fn(pred, occ)
            total_loss += float(loss)
            total_points += hyperparams["batch_size"]
            total_correct += torch.sum(torch.where(pred > 0.5, 1, 0) == occ)
        print(f"Average 2D Testing Loss: {total_loss/total_points}\n")
        print(f"2D Testing Accuracy: {total_correct/total_points}\n")
        experiment.log_metric("2d loss", total_loss/total_points)
        experiment.log_metric("2d accuracy", total_correct/total_points)

def test_3d(model, test_loader, experiment):
    '''
    Tests the model on validation data
    '''
    loss_fn = nn.BCELoss()
    model = model.eval()
    with torch.no_grad(), experiment.test():
        total_loss = 0
        total_points = 0
        total_correct = 0
        for uv, xyz,occ, cam_params in tqdm(test_loader):
            uv = uv.to(device)
            xyz = xyz.to(device)
            occ = occ.to(device)
            cam_params = cam_params.to(device)
            pred = model(uv, cam_params, xyz=xyz).reshape((hyperparams["batch_size"],))
            loss = loss_fn(pred, occ)
            total_loss += float(loss)
            total_points += hyperparams["batch_size"]
            total_correct += torch.sum(torch.where(pred > 0.5, 1, 0) == occ)
        print(f"Average 3D Testing Loss: {total_loss/total_points}\n")
        print(f"3D Testing Accuracy: {total_correct/total_points}\n")
        experiment.log_metric("3d loss", total_loss/total_points)
        experiment.log_metric("3d accuracy", total_correct/total_points)

def visualize_2d(model, obj, cam_center, show=True, log=False, experiment=None, epoch=None):
    print("Visualizing 2D Occupancy...")
    model = model.eval()
    with torch.no_grad():
        l, x, y, _, _ = obj.contains_2d(cam_center, hyperparams["focal_length"], hyperparams["sensor_bounds"], hyperparams["sensor_size"])
        x_batch = np.reshape(x, (-1,1))
        y_batch = np.reshape(y, (-1,1))
        coord_batch = torch.hstack([torch.tensor(x_batch, dtype=torch.float32), torch.tensor(y_batch, dtype=torch.float32)])
        r = rotation_matrix(cam_center, inverse=True).flatten()
        cam_params = list(r) + cam_center
        params_batch = torch.tensor([cam_params]*coord_batch.shape[0], dtype=torch.float32)
        coord_batch = coord_batch.to(device)
        params_batch = params_batch.to(device)
        model_labels = model.forward(coord_batch, params_batch)
        model_labels = np.reshape(model_labels.cpu().numpy(), l.shape)

        # show results
        f = plt.figure()
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)
        ax1.imshow(l)
        ax1.set_title("Ground Truth")
        ax2.imshow(model_labels)
        ax2.set_title("2D Occ Network")
        
        if show:
            plt.show()
        if log:
            if experiment is None:
                print("Must provide experiment to visualizer in order to log figure")
            else:
                experiment.log_figure(figure=f, figure_name=f"epoch_{epoch}_2d")
                plt.close(f)

def visualize_3d(model, obj, bbox, num_points=1000, samples_per_camera=10000, show=True, log=False, experiment=None, epoch=None):
    '''
    Visualizes the learned occupancy of the model as well as the 
    '''
    print("Visualizing 3D Occupancy...")
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    # TODO: show ground truth
    # TODO: add uniform bbox sampling method in utils

    ((x_min, x_max),(y_min, y_max),(z_min, z_max)) = bbox

    points_to_plot = []
    attempts = 0
    pbar = tqdm(total=num_points)
    while(len(points_to_plot) < num_points and attempts < 200):
        attempts += 1
        # TODO: abstract out this functionality so it isn't copied from data.py
        cam_center = random_cam_center(hyperparams["cam_radius"])
        _, u, v, occ_3d, coords_3d = obj.contains_2d(cam_center, hyperparams["focal_length"], hyperparams["sensor_bounds"], hyperparams["sensor_size"], ray_samples=20)
        coords_3d = np.reshape(coords_3d, (-1,3))
        occ_3d = occ_3d.flatten()
        u = np.stack([u]*5, axis=-1).flatten()
        v = np.stack([v]*5, axis=-1).flatten()
        r = rotation_matrix(cam_center, inverse=True).flatten()
        cam_params = list(r) + cam_center

        labels = []
        xyz = []
        uv = []
        cp = []

        # samples some fraction of the pixels to add to our data
        # for i in random.sample(range(coords_3d.shape[0]), samples_per_camera):
        for i in range(coords_3d.shape[0]):
            labels.append(torch.tensor(occ_3d[i], dtype=torch.float32))
            xyz.append(torch.tensor(coords_3d[i], dtype=torch.float32))
            uv.append(torch.tensor([u[i], v[i]], dtype=torch.float32))
            cp.append(torch.tensor(cam_params, dtype=torch.float32))

        labels = torch.vstack(labels)
        xyz = torch.vstack(xyz)
        uv= torch.vstack(uv)
        cp = torch.vstack(cp)
        print(xyz)

        occ = model(uv, cp, xyz=xyz)

        prev_points = len(points_to_plot)        
        for i in range(occ.shape[0]):
            if occ[i] > 0.5:
                points_to_plot.append(xyz[i])
        pbar.update(len(points_to_plot) - prev_points)
    tqdm._instances.clear()

    
    xs = [p[0] for p in points_to_plot]
    ys = [p[1] for p in points_to_plot]
    zs = [p[2] for p in points_to_plot]

    ax1.scatter(xs, ys, zs)
    ax1.set_xlim3d(left=x_min, right=x_max)
    ax1.set_ylim3d(bottom=y_min, top=y_max)
    ax1.set_zlim3d(bottom=z_min, top=z_max)
    if show:
        plt.show()
    if log:
        if experiment is None:
            print("Must provide experiment to visualizer in order to log figure")
        else:
            experiment.log_figure(figure=fig, figure_name=f"epoch_{epoch}_3d")
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the 2D neural body model")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-v", "--viz", action="store_true",
                        help="visualize the model and gt occupancy")
    parser.add_argument("-n", "--num_samples", default=8192, type=int,
                        help="number of sampled points for training")
    args = parser.parse_args()

    # Log to comet ml
    experiment = Experiment(log_code=False)
    experiment.log_parameters(hyperparams)

    # make cuboid object to learn occupancy of
    obj = Cuboid((0,0,0), 0.1, 0.3, 0.8)
    bbox = ((-1.,1.), (-1.,1.), (-1.,1.))

    # make model
    model = UnstructuredHybrid(num_layers=hyperparams["layers"], hidden_size=hyperparams["hidden_size"]).to(device)

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./saved_models/modelhybrid.pt'))
    if args.train:
        print("running training loop...")
        train_data_2d = OccDataset2D(obj, args.num_samples, hyperparams["focal_length"], hyperparams["sensor_bounds"], hyperparams["sensor_size"], hyperparams["cam_radius"])
        train_data_3d = OccDataset3DUVD(obj, args.num_samples*hyperparams["3d_ratio"], hyperparams["focal_length"],hyperparams["sensor_bounds"], hyperparams["sensor_size"], hyperparams["cam_radius"])
        train_loader_2d = DataLoader(train_data_2d, batch_size=hyperparams["batch_size"], shuffle=True, drop_last=True)
        train_loader_3d = DataLoader(train_data_3d, batch_size=hyperparams["batch_size"], shuffle=True, drop_last=True)
        train(model, train_loader_2d, train_loader_3d, experiment)
    if args.test:
        print("running testing loop...")
        test_data_2d = OccDataset2D(obj, args.num_samples, hyperparams["focal_length"], hyperparams["sensor_bounds"], hyperparams["sensor_size"], hyperparams["cam_radius"])
        test_data_3d = OccDataset3DUVD(obj, args.num_samples, hyperparams["focal_length"], hyperparams["sensor_bounds"], hyperparams["sensor_size"], hyperparams["cam_radius"])
        test_loader_2d = DataLoader(test_data_2d, batch_size=hyperparams["batch_size"], shuffle=True, drop_last=True)
        test_loader_3d = DataLoader(test_data_3d, batch_size=hyperparams["batch_size"], shuffle=True, drop_last=True)
        test_2d(model, test_loader_2d, experiment)
        test_3d(model, test_loader_3d, experiment)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './saved_models/modelhybrid.pt')
    if args.viz:
        print("visualizing occupancy...")
        visualize_2d(model, obj, random_cam_center(hyperparams["cam_radius"]))
        visualize_3d(model, obj, bbox)

