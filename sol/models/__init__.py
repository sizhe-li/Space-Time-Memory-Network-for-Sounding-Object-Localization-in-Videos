import cv2
import math
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from scipy import ndimage
from torchvision.ops.boxes import box_iou
from .audio_nets import Unet, ANet, VGG, VGGish
from .video_nets import Resnet, VGG16

def find_bbox_in_heatmap(prob_map, size, thresh=0.5):
    """
    Args:
        prob_map (np.ndarry) (HxW)
        size = (tuple)
    """
    prob_map[prob_map < thresh] = 0
    prob_map = np.uint8(prob_map * 255)

    xtl, ytl, xbr, ybr = math.inf, math.inf, -math.inf, -math.inf
    thresh = cv2.threshold(prob_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) >= 100]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        xtl, ytl = min(xtl, x), min(ytl, y)
        xbr, ybr = max(xbr, x + w), max(ybr, y + h)

    return np.array([xtl, ytl, xbr, ybr], dtype=np.float32)


def calc_iou(pred_box, gt_box):
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of union
    uni = (
        (pred_box[2] - pred_box[0] + 1.0) * (pred_box[3] - pred_box[1] + 1.0)
        + (gt_box[2] - gt_box[0] + 1.0) * (gt_box[3] - gt_box[1] + 1.0)
        - inters
    )

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni
    return iou


def calc_ciou(location, gtmap, thresh):
    assert location.shape == gtmap.shape
    ciou = np.sum(gtmap[location > thresh]) / (
        np.sum(gtmap) + np.sum(gtmap[location > thresh] == 0)
    )
    return ciou


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ModelBuilder:
    def build_sound(self, arch="vggish", fc_dim=64, weights=""):
        # 2D models
        if arch == "unet5":
            net_sound = Unet(fc_dim=fc_dim, num_downs=5)
        elif arch == "unet6":
            net_sound = Unet(fc_dim=fc_dim, num_downs=6)
        elif arch == "unet7":
            net_sound = Unet(fc_dim=fc_dim, num_downs=7)
        elif arch == "vggish":
            net_sound = VGGish()
        elif arch == "anet":
            net_sound = ANet()
        elif arch == "vgg16":
            pretrained = True
            original_vgg16 = torchvision.models.vgg16(pretrained)
            net_sound = VGG(original_vgg16)
        else:
            raise Exception("Architecture undefined!")

        # net_sound.apply(self.weights_init)
        if len(weights) > 0:
            print("Loading weights for net_sound")
            net_sound.load_state_dict(torch.load(weights))

        return net_sound

    # builder for vision
    def build_video(
        self,
        arch="resnet50",
        fc_dim=64,
        pool_type="avgpool",
        weights="",
        pretrained=True,
    ):
        if arch == "vgg16":
            original_vgg16 = torchvision.models.vgg16(pretrained)
            net = VGG16(original_vgg16, init_weights=False)
        elif arch.startswith("resnet"):
            net = Resnet(torchvision.models.__dict__[arch](pretrained))
        else:
            raise Exception("Architecture undefined!")

        if len(weights) > 0:
            print("Loading weights for net_frame")
            net.load_state_dict(torch.load(weights))
        return net
