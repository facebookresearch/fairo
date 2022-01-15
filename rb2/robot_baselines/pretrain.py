import argparse
import baselines
import torch, os
import numpy as np
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt


# parse inputs and constants
parser = argparse.ArgumentParser()
parser.add_argument('output_folder')
parser.add_argument('--input_data', default='./data/pretext.npz')
parser.add_argument('--net', default='VGGSoftmax')
parser.add_argument('--BATCH_SIZE', default=32, type=int)
parser.add_argument('--EPOCHS', default=100, type=int)
parser.add_argument('--LR', default=1e-4, type=float)
args = parser.parse_args()
train, test, mean_train_pos = baselines.datasets.pretext_dataset(args.input_data, args.BATCH_SIZE)
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)
else:
    input('Path already exists! Press enter to continue')


# build network and restore weights
features = baselines.get_network(args.net)
baselines.net.restore_pretrain(features)
point_pred = baselines.net.PointPredictor(256, mean_train_pos)
net = nn.Sequential(features, point_pred)
net = net.cuda().train()


# build optim
train_metric, test_metric = baselines.Metric(), baselines.Metric()
optim = torch.optim.Adam(net.parameters(), lr=args.LR)
loss = nn.SmoothL1Loss()
for e in range(args.EPOCHS):
    train_metric.reset(); test_metric.reset()
    net = net.train()
    for img, pos in train:
        optim.zero_grad()
        pos_hat = net(img.cuda())
        train_loss = loss(pos_hat, pos.cuda())
        train_loss.backward()
        optim.step()
        train_metric.add(train_loss.item())
    
    net = net.eval()
    for img, pos in test:
        with torch.no_grad():
            pos_hat = net(img.cuda())
            test_metric.add(loss(pos_hat, pos.cuda()).item())
        ph, p = pos_hat.cpu().numpy(), pos.numpy()

    if e % 10 == 0:
        plt.close()
        plt.scatter(ph[-15:,0], ph[-15:,1], color='red')
        plt.scatter(p[-15:,0],  p[-15:,1],  color='blue')
        for pred, real in zip(ph[-15:], p[-15:]):
            plt.plot([pred[0], real[0]], [pred[1], real[1]], color='red')
        plt.savefig('{}/compar{}.jpg'.format(args.output_folder, e))

    print('epoch {} \t train {:.6f} \t test {:.6f}'.format(e, train_metric.mean, test_metric.mean))


net = net.cpu()
torch.save(features.state_dict(), args.output_folder + '/features.pt')
torch.save(point_pred.state_dict(), args.output_folder + '/point_pred.pt')
