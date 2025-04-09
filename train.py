import os
import argparse
import copy
import yaml
import time
import glob

import torch
import torch.optim as optim
import torch.utils as utils
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import albumentations
import albumentations.pytorch

from femoral_datasets import FemoralDataset, make_batch
from general import plot_image_from_output
from utils_ObjectDetection import *
from val import val

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def args_config():
    parser = argparse.ArgumentParser(description='Object Detection for Femoral Artery Project')
    parser.add_argument('-e', '--epochs', default=100, type=int,
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--train_batch', default=16, type=int,
                        help='train batch size (default: 128)')
    parser.add_argument('--val_batch', default=8, type=int,
                        help='val batch size (default: 32)')
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float,
                        help='initial Learning rate (default: 0.01)')
    parser.add_argument('-m', '--momentum', default=0.9, type=float,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('-w', '--weight_decay', default=0, type=float)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--result', default='./runs/train', type=str)
    parser.add_argument('--exp', default='exp', type=str)

    return parser.parse_args()

def train(model, dataloader, optimizer, scheduler, device):

    model.train()

    epoch_loss = 0
    losses_dict = dict()
    idx = 1
    for imgs, targets in dataloader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
        if idx == 1:
            losses_dict = {k: v for k, v in loss_dict.items()}
        else:
            losses_dict = {k: v+losses_dict[k] for k, v in loss_dict.items()}

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses
        idx += 1

    scheduler.step()
    losses_dict = {k: v/idx for k, v in losses_dict.items()}
    return epoch_loss, losses_dict


def main():
    best_mAP = 0
    args = args_config()
    print (args)

    path = f'{args.result}'
    if not os.path.isdir(path):
        os.mkdir(path)

    list_dir = os.listdir(path)
    if len(list_dir) == 0:
        save_dir = f'{args.result}/{args.exp}0'
        os.mkdir(save_dir)

    else:
        save_dir = f'{args.result}/{args.exp}{len(list_dir)}'
        os.mkdir(save_dir)

    """
    albumentations_transform = albumentations.Compose([
        albumentations.OneOf([
            albumentations.HorizontalFlip(p=1),
            albumentations.RandomRotate90(p=1),
            albumentations.VerticalFlip(p=1)
        ], p=1),
        albumentations.OneOf([
            albumentations.MotionBlur(p=1),
            albumentations.GaussNoise(p=1)
        ], p=1),
        albumentations.pytorch.transforms.ToTensorV2()
    ], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels']))


    train_dataloader = data.DataLoader(
        FemoralDataset('./data',
         albumentations_transform)
        ,batch_size=args.train_batch, shuffle=True, num_workers=4, collate_fn=make_batch
    )

    val_dataloader = data.DataLoader(
        FemoralDataset('./data',
                       transforms=albumentations.Compose([
                           albumentations.pytorch.transforms.ToTensorV2()
                       ], bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels'])),
                       dataset='val')
        ,batch_size=args.val_batch, shuffle=False, num_workers=4, collate_fn=make_batch
    )
    """
    tf = transforms.Compose([transforms.ToTensor()
                             ])
    train_dataloader = data.DataLoader(
        FemoralDataset('./data',
                       tf)
        , batch_size=args.train_batch, shuffle=True, num_workers=4, collate_fn=make_batch
    )

    val_dataloader = data.DataLoader(
        FemoralDataset('./data',
                       tf,
                       dataset='val')
        , batch_size=args.val_batch, shuffle=False, num_workers=4, collate_fn=make_batch
    )


    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, args.num_classes)


    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print ('===== Start Training =====')
    for epoch in range(args.epochs):
        print (f'===== Epochs: {epoch+1}/{args.epochs} =====')
        start = time.time()
        losses, losses_dict = train(model, train_dataloader, optimizer, scheduler, device)
        print (f'losses = {losses}, time = {time.time()-start}')

        preds_adj_all, annot_all, labels = val(model, val_dataloader, device, conf_threshold=0.001)

        sample_metrics = []
        for batch_i in range(len(preds_adj_all)):
            sample_metrics += get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.6)

        true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, torch.tensor(labels))
        print (AP)
        mAP = torch.mean(AP)

        print (f'mAP : {mAP}')
        print (f'AP : {AP}')


        with open(f'{save_dir}/train_losses.txt', 'a') as f:
            f.write(f'{losses}\n')
        with open(f'{save_dir}/train_losses_dict.txt', 'a') as f:
            f.write(f'{losses_dict}\n')

        if mAP > best_mAP:
            best_mAP = mAP
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, f'{save_dir}/best.pt')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, f'{save_dir}/model.pt')

if __name__ == '__main__':
    main()