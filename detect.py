import os
import argparse
import copy
import yaml
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt

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
from segmentation import segmentation

#from utils_ObjectDetection import *

classes = ['background', 'stenosis', 'occlusion', 'stent']

def main():
    img_path = './data/detect'

    images = list(sorted(os.listdir(img_path)))
    tf = transforms.Compose([
        transforms.ToTensor()
    ])

    save_dir = './runs/detect/exp0'
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

    checkpoint = torch.load('./runs/train/exp0/best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    model.eval()

    for image in images:
        img_file = f'{img_path}/{image}'
        img = Image.open(img_file).convert('RGB')
        img_t = tf(img)
        img_t = [img_t.to(device)]
        with torch.no_grad():
            start_time = time.time()
            pred_adj = model(img_t)
            print (time.time() - start_time)
            idx_list = []
            for idx, score in enumerate(pred_adj[0]['scores']):
                if score > 0.25:
                    idx_list.append(idx)

            pred_adj[0]['boxes'] = pred_adj[0]['boxes'][idx_list]
            pred_adj[0]['labels'] = pred_adj[0]['labels'][idx_list]
            pred_adj[0]['scores'] = pred_adj[0]['scores'][idx_list]

            pred_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in pred_adj]

            img = segmentation(np.array(img))
            img = Image.fromarray(img)
            plot_image_from_output(img, pred_adj[0], save_dir, image)

def plot_image_from_output(img, annotation, save_dir, img_file):
    draw = ImageDraw.Draw(img)
    xmin, ymin, xmax, ymax = annotation["boxes"][0]
    draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=(0, 0, 255), width=3)

    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]
        label = classes[annotation['labels'][idx]]
        score = annotation['scores'][idx]
        text = f'{label} {score:.2f}'
        font= ImageFont.truetype('arial.ttf', 20)
        if annotation['labels'][idx] == 1:
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=(255, 0, 0), width=4)
            draw.rectangle(((xmin, ymin-25), (xmin+120, ymin)), outline=(255, 0, 0), fill=(255, 0, 0))
            draw.text((xmin, ymin-23), text, font=font, fill=(255, 255, 255))

        elif annotation['labels'][idx] == 2:
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=(255, 80, 150), width=4)
            draw.rectangle(((xmin, ymin - 25), (xmin + 160, ymin)), outline=(255, 80, 150), fill=(255, 80, 150))
            draw.text((xmin, ymin - 23), text, font=font, fill=(255, 255, 255))

        elif annotation['labels'][idx] == 3:
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=(255, 150, 0), width=4)
            draw.rectangle(((xmin, ymin - 25), (xmin + 120, ymin)), outline=(255, 150, 0), fill=(255, 150, 0))
            draw.text((xmin, ymin - 23), text, font=font, fill=(255, 255, 255))


    img.save(f'{save_dir}/{img_file}')

if __name__ == '__main__':
    main()