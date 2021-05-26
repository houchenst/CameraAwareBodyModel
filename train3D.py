'''
Trains the 3D occupancy model
'''

from torch import nn
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from utils import Cuboid
from models import Unstructured3D
from data import OccDataset3D

# NASA paper uses batch size 12, learning rate 1e-4
hyperparams = {
    "batch_size" : 12,
    "learning_rate" :  1e-4,
    "epochs" : 100,
}
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
device = "cpu"
# print(f"{torch.cuda.get_device_properties(device).total_memory}\n\n")
# print(f"{torch.cuda.memory_reserved(device)}\n\n")
# print(f"{torch.cuda.memory_allocated(device)}\n\n")

def train(model, train_loader, hyperparams):
    '''
    Trains the model
    '''
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    model = model.train()
    for e in range(hyperparams["epochs"]):
        total_loss = 0
        total_points = 0
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x).reshape((hyperparams["batch_size"],))
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss
            total_points += hyperparams["batch_size"]
        print(f"Epoch {e} Average Training Loss: {total_loss/total_points}\n")



def test(model, test_loader):
    '''
    Tests the model on validation data
    '''
    loss_fn = nn.BCELoss()

    model = model.eval()
    with torch.no_grad():
        total_loss = 0
        total_points = 0
        total_correct = 0
        for x,y in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x).reshape((hyperparams["batch_size"],))
            loss = loss_fn(pred, y)
            total_loss += loss
            total_points += hyperparams["batch_size"]
            total_correct += torch.sum(torch.where(pred > 0.5, 1, 0) == y)
        print(f"Average Testing Loss: {total_loss/total_points}\n")
        print(f"Testing Accuracy: {total_correct/total_points}\n")

def visualize(model, obj, bbox, num_points=1000):
    '''
    Visualizes the learned occupancy of the model as well as the 
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    # TODO: show ground truth
    # TODO: add uniform bbox sampling method in utils

    ((x_min, x_max),(y_min, y_max),(z_min, z_max)) = bbox

    points_to_plot = []
    batching = 1000
    while(len(points_to_plot) < num_points):
        points = [[random.uniform(x_min, x_max), random.uniform(y_min, y_max), random.uniform(z_min, z_max)] for _ in range(batching)]
        # for p in points:
        #     if obj.contains(p):
        #         points_to_plot.append(p)
        pred = model(torch.tensor(points)).reshape((batching,))
        for i in range(pred.shape[0]):
            if pred[i] > 0.5:
                points_to_plot.append(points[i])
    
    xs = [p[0] for p in points_to_plot]
    ys = [p[1] for p in points_to_plot]
    zs = [p[2] for p in points_to_plot]

    ax1.scatter(xs, ys, zs)
    ax1.set_xlim3d(left=x_min, right=x_max)
    ax1.set_ylim3d(bottom=y_min, top=y_max)
    ax1.set_zlim3d(bottom=z_min, top=z_max)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the neural body model")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-v", "--viz", action="store_true",
                        help="visualize the model and gt occupancy")
    parser.add_argument("-n", "--num_samples", default=8192, type=int,
                        help="number of sampled points for training")
    args = parser.parse_args()

    # make cuboid object to learn occupancy of
    obj = Cuboid((0,0,0), 0.1, 0.3, 0.8)
    bbox = ((-1.,1.), (-1.,1.), (-1.,1.))

    # make model
    model = Unstructured3D().to(device)

    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt'))
    if args.train:
        print("running training loop...")
        train_data = OccDataset3D(obj, bbox, args.num_samples)
        train_loader = DataLoader(train_data, batch_size=hyperparams["batch_size"], shuffle=True, drop_last=True)
        train(model, train_loader, hyperparams)
    if args.test:
        print("running testing loop...")
        test_data = OccDataset3D(obj, bbox, args.num_samples//4)
        test_loader = DataLoader(test_data, batch_size=hyperparams["batch_size"], shuffle=True, drop_last=True)
        test(model, test_loader)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')
    if args.viz:
        print("visualizing occupancy...")
        visualize(model, obj, bbox)

