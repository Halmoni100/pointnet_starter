import argparse
from functools import partial

import torch

from Pointnet_Pointnet2_pytorch.models import pointnet_cls

from pointnet_dataset import PointNetDataset
from torch_helper import train

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', '-d', type=str, help='Directory for .pcd files')
args = parser.parse_args()

batch_size = 24
train_ratio = 0.8
num_categories = 2
num_epochs = 200
learning_rate = 0.001
num_points = 1024
decay_rate = 1e-4

total_dataset = PointNetDataset(args.datadir)
total_size = len(total_dataset)
train_size = int(train_ratio * total_size)
val_size = total_size = train_size
train_dataset, val_dataset = torch.utils.data.random_split(total_dataset, [train_size, val_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

model = pointnet_cls.get_model(num_categories)
model_loss_func = model.get_loss()
def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
model.apply(inplace_relu)

model = model.cuda()
model_loss_func = model_loss_func.cuda()

def criterion(outputs, labels):
    pred = outputs[0]
    trans_feat = outputs[1]
    return model_loss_func(pred, labels.long(), trans_feat)

def metric(outputs, labels):
    pred = outputs[0]
    pred_choice = pred.data.max(1)[1]
    return pred_choice.eq(labels.long().data).cpu().sum()

def inputs_labels_func(data):
    points, labels = data
    points = torch.Tensor(points)
    points = points.transpose(2, 1)
    labels = torch.Tensor(labels)
    return points, labels

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=decay_rate
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

train.train(train_dataloader, val_dataloader, train_size, val_size, inputs_labels_func,
            model, criterion, optimizer, metric=metric, metric_name="accuracy", scheduler=scheduler,
            device=torch.cuda, num_epochs=num_epochs, do_carriage_return=False)
