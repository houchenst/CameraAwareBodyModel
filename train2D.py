from comet_ml import Experiment
from torch import nn
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import random

from utils import *
from models import Unstructured2D
from data import OccDataset2D

# NASA paper uses batch size 12, learning rate 1e-4
hyperparams = {
    "batch_size" : 12,
    "learning_rate" :  1e-4,
    "epochs" : 10,
    "layers" : 8,
    "hidden_size": 80,

    "focal_length": 2.0,
    "sensor_bounds": [0.5, 0.5],
    "sensor_size": [100, 100],
    "cam_radius": 3.0,
}
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
device = "cpu"
# print(f"{torch.cuda.get_device_properties(device).total_memory}\n\n")
# print(f"{torch.cuda.memory_reserved(device)}\n\n")
# print(f"{torch.cuda.memory_allocated(device)}\n\n")

def train(model, train_loader, experiment):
    '''
    Trains the model
    '''
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    model = model.train()
    with experiment.train():
        for e in range(hyperparams["epochs"]):
            total_loss = 0
            total_points = 0
            for x, y, cam_params in tqdm(train_loader):
                x = x.to(device)
                y = y.to(device)
                cam_params = cam_params.to(device)
                pred = model(x, cam_params).reshape((hyperparams["batch_size"],))
                optimizer.zero_grad()
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss)
                total_points += hyperparams["batch_size"]
            if e % 10 == 0:
                print(f"Epoch {e} Average Training Loss: {total_loss/total_points}\n")



def test(model, test_loader, experiment):
    '''
    Tests the model on validation data
    '''
    loss_fn = nn.BCELoss()

    model = model.eval()
    with torch.no_grad(), experiment.test():
        total_loss = 0
        total_points = 0
        total_correct = 0
        for x,y, cam_params in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device)
            cam_params = cam_params.to(device)
            pred = model(x, cam_params).reshape((hyperparams["batch_size"],))
            loss = loss_fn(pred, y)
            total_loss += float(loss)
            total_points += hyperparams["batch_size"]
            total_correct += torch.sum(torch.where(pred > 0.5, 1, 0) == y)
        print(f"Average Testing Loss: {total_loss/total_points}\n")
        print(f"Testing Accuracy: {total_correct/total_points}\n")
        experiment.log_metric("loss", total_loss/total_points)
        experiment.log_metric("accuracy", total_correct/total_points)

def visualize(model, obj, cam_center):
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
        plt.show()


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
    model = Unstructured2D(num_layers=hyperparams["layers"], hidden_size=hyperparams["hidden_size"]).to(device)

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./saved_models/model2d.pt'))
    if args.train:
        print("running training loop...")
        train_data = OccDataset2D(obj, args.num_samples, hyperparams["focal_length"], hyperparams["sensor_bounds"], hyperparams["sensor_size"], hyperparams["cam_radius"])
        train_loader = DataLoader(train_data, batch_size=hyperparams["batch_size"], shuffle=True, drop_last=True)
        train(model, train_loader, experiment)
    if args.test:
        print("running testing loop...")
        test_data = OccDataset2D(obj, args.num_samples//4, hyperparams["focal_length"], hyperparams["sensor_bounds"], hyperparams["sensor_size"], hyperparams["cam_radius"])
        test_loader = DataLoader(test_data, batch_size=hyperparams["batch_size"], shuffle=True, drop_last=True)
        test(model, test_loader, experiment)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './saved_models/model2d.pt')
    if args.viz:
        print("visualizing occupancy...")
        visualize(model, obj, random_cam_center(hyperparams["cam_radius"]))

