from __future__ import division

# from models import *
from tool.darknet2pytorch import *
from utils.augmentation import *
from utils.datasets import *
from utils.utils import *
from utils.region_loss import *
from tool.cfg import *

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="cfg/yolov4.cfg", help="path to model definition file")
    parser.add_argument("--pretrained_weights", type=str,default="weight/yolov4.weights",help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=608, help="size of each image dimension")
    parser.add_argument("--data_root", type=str, default="/disk_d/workspace/personalSpace/like_project/VOC_/VOCdevkit/VOC2007",
                        help="The data directory")
    parser.add_argument("--cfgfile", type=str, default="cfg/yolov4.cfg",
                        help="The cfgfile directory")
    opt = parser.parse_args()
    net_opt = parse_cfg(opt.cfgfile)

    # Initiate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def,is_train=True).to(device)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_weights(opt.pretrained_weights)

    # Get dataloader
    train_dataset = VOCDetection(root=opt.data_root)
    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batch_size,
                              num_workers=2,
                              shuffle=True,
                              collate_fn=collater,
                              pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters())

    # region_loss = RegionLoss()
    # loss_opt = net_opt[-1]
    # anchors = loss_opt['anchors'].split(',')
    # anchor_mask = loss_opt['mask'].split(',')
    # region_loss.anchor_mask = [int(i) for i in anchor_mask]
    # region_loss.anchors = [float(i) for i in anchors]
    # region_loss.num_classes = int(loss_opt['classes'])
    # region_loss.num_anchors = int(loss_opt['num'])
    # region_loss.anchor_step = len(region_loss.anchors) // region_loss.num_anchors

    mode_path = r'weight/net.pth'
    mode_save = r'weight/net.pth'
    model.load_state_dict(torch.load(mode_save))
    # model.load_weights(mode_path)
    for epoch in range(opt.epochs):
        model.train()
        for batch_i, (imgs, targets) in enumerate(train_loader):

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            optimizer.zero_grad()
            loss = model(imgs,targets)

            # loss = region_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            print(epoch,loss.item())

    torch.save(model.state_dict(), mode_save)




