import os
import argparse
import copy
import yaml
import time

from threading import Thread

import torch
import torch.optim as optim
import torch.utils as utils
import torch.utils.data as data


import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from femoral_datasets import FemoralDataset, make_batch
from general import plot_image_from_output
from utils_ObjectDetection import *

from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt

def args_config():
    parser = argparse.ArgumentParser(description='Object Detection for Femoral Artery Project')
    parser.add_argument('--val_batch', default=4, type=int,
                        help='val batch size (default: 32)')
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--result', default='./runs/val', type=str)
    parser.add_argument('--exp', default='exp', type=str)
    parser.add_argument('--conf_thres', default=0.001, type=float)
    parser.add_argument('--iou_thres', default=0.6, type=float)

    return parser.parse_args()

def val(model, dataloader, device, conf_threshold=0.001):
    model.eval()

    labels = []
    preds_adj_all = []
    annot_all = []

    for imgs, targets in dataloader:
        imgs = list(img.to(device) for img in imgs)

        for t in targets:
            labels += t['labels']

        with torch.no_grad():
            preds_adj = model(imgs)
            for id in range(len(preds_adj)):

                idx_list = []

                for idx, score in enumerate(preds_adj[id]['scores']):
                    if score > conf_threshold:
                        idx_list.append(idx)

                preds_adj[id]['boxes'] = preds_adj[id]['boxes'][idx_list]
                preds_adj[id]['labels'] = preds_adj[id]['labels'][idx_list]
                preds_adj[id]['scores'] = preds_adj[id]['scores'][idx_list]

            preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
            preds_adj_all.append(preds_adj)
            annot_all.append(targets)

    return preds_adj_all, annot_all, labels

def main():
    args = args_config()
    print(args)

    val_dataloader = data.DataLoader(
        FemoralDataset('./data',
                       transforms.ToTensor(),
                       dataset='val'),
        batch_size=args.val_batch, shuffle=False, num_workers=4, collate_fn=make_batch
    )

    save_dir = './runs/val/exp0'
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, args.num_classes)

    checkpoint = torch.load('./runs/train/exp0/best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    preds_adj_all, annot_all, labels = val(model, val_dataloader, device, conf_threshold=args.conf_thres)

    sample_metrics = []
    for batch_i in range(len(preds_adj_all)):
        sample_metrics += get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=args.iou_thres)


    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]


    precision, recall, AP, f1, ap_class, pc, rc = ap_per_class(true_positives, pred_scores, pred_labels, torch.tensor(labels), save_dir)
    fig, ax = plt.subplots(figsize=(6, 5))
    viz = PrecisionRecallDisplay(
        precision=pc[0],
        recall=rc[0],
        average_precision=AP[0]
    )
    viz.plot(ax=ax, name='stenosis')

    viz = PrecisionRecallDisplay(
        precision=pc[1],
        recall=rc[1],
        average_precision=AP[1]
    )
    viz.plot(ax=ax, name='occlusion')
    
    viz = PrecisionRecallDisplay(
        precision=pc[2],
        recall=rc[2],
        average_precision=AP[2]
    )
    viz.plot(ax=ax, name='stent')


    mAP = torch.mean(AP)
    
    print ('precision', precision)
    print ('recall', recall)
    print (f1)
    print (ap_class)
    print ('ap', AP)
    print (mAP)

    plt.show()


if __name__ == '__main__':
    main()
