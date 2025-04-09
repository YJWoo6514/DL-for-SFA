import os
from PIL import Image
import cv2

import torch
import torchvision
import torchvision.transforms as transforms

class FemoralDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, dataset='train'):
        self.root = root
        self.transforms = transforms

        self.imgs_path = os.path.join(self.root, f'images/{dataset}')
        self.masks_path = os.path.join(self.root, f'labels(rcnn)/{dataset}')

        self.imgs = list(sorted(os.listdir(self.imgs_path)))
        self.masks = list(sorted(os.listdir(self.masks_path)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs_path, self.imgs[idx])
        mask_path = os.path.join(self.masks_path, self.masks[idx])
        img = Image.open(img_path).convert('RGB')
        #img = cv2.imread(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        with open(mask_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                line = line.split(' ')

                labels.append(int(line[0]))
                box = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
                boxes.append(box)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        targets = {'boxes': boxes,
                   'labels': labels}

        """
        if self.transforms:
            transformed = self.transforms(image = img, bboxes = boxes, labels = labels)
            img = torch.as_tensor(transformed['image'], dtype=torch.float32)
            targets = {'boxes':torch.as_tensor(transformed['bboxes'], dtype=torch.float32),
                       'labels':torch.as_tensor(transformed['labels'], dtype=torch.int64)}


        """
        if self.transforms:
            img = self.transforms(img)

        return img, targets

    def __len__(self):
        return len(self.imgs)


def make_batch(samples):
    """
    imgs = [sample[0] for sample in samples]
    #boxes = torch.nn.utils.rnn.pad_sequence([sample[1]['boxes'] for sample in samples],
    #                                               batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence([sample[1]['labels'] for sample in samples],
                                                    batch_first=True)
    boxes = []
    print (boxes)
    return imgs, {'boxes': boxes.contiguous(), 'labels': labels.contiguous()}
    """
    return tuple(zip(*samples))