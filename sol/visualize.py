import os
import math
import shutil
import cv2
import numpy as np

import torch
import torch.nn as nn

from torchvision import transforms as tf
from torch.utils.data import DataLoader
from tqdm import tqdm
from moviepy.editor import AudioFileClip, ImageSequenceClip
from arguments import ArgParser
from dataset import AVEDataset, MusicDataset


from models.stm_nets import STM

time_length = 10

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
# fontScale
fontScale = 0.5
# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

# helpers
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        T = []
        for t, m, s in zip(tensor, self.mean, self.std):
            T.append(t.mul(s).add(m))
            # The normalize code -> t.sub_(m).div_(s)
        return torch.stack(T)


class WrappedModel(nn.Module):
    def __init__(self, module):

        super(WrappedModel, self).__init__()
        self.module = module  # that I actually define.

    def forward(self, x):

        return self.module(x)


def apply_heatmap(img, prob_map, size, bbox=False, thresh=0.5):
    """
    Args:
        image (PIL.Image)
        prob_map (numpy.ndarray)
        size (length2 tuple)
    """
    #     prob_map = prob_map
    prob_map = np.maximum(prob_map, 0)
    prob_map = cv2.resize(prob_map, size)
    prob_map = prob_map - np.min(prob_map)
    prob_map = prob_map / np.max(prob_map)
    prob_map[prob_map < thresh] = 0

    gray = np.uint8(prob_map * 255)
    prob_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    prob_map = cv2.cvtColor(prob_map, cv2.COLOR_BGR2RGB)
    prob_map = np.float32(prob_map) / 255

    cam = prob_map + np.float32(img) / 255
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
def apply_heatmap(img, prob_map, size, bbox=False, thresh=0.5):
    """
    Args:
        image (PIL.Image)
        prob_map (numpy.ndarray)
        size (length2 tuple)
    """
    #     prob_map = prob_map
    prob_map = np.maximum(prob_map, 0)
    prob_map = cv2.resize(prob_map, size)
    prob_map = prob_map - np.min(prob_map)
    prob_map = prob_map / np.max(prob_map)
    prob_map[prob_map < thresh] = 0

    gray = np.uint8(prob_map * 255)
    prob_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    prob_map = cv2.cvtColor(prob_map, cv2.COLOR_BGR2RGB)
    prob_map = np.float32(prob_map) / 255

    cam = prob_map + np.float32(img) / 255
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)

    if bbox:
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) >= 1000]
        # xtl, ytl, xbr, ybr = math.inf, math.inf, -math.inf, -math.inf
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(cam, (x, y), (x + w, y + h), (36, 255, 12), 2)
            # xtl, ytl = min(xtl, x), min(ytl, y)
            # xbr, ybr = max(xbr, x + w), max(ybr, y + h)
        # cv2.rectangle(cam, (xtl, ytl), (xbr, ybr), (36, 255, 12), 2)
    return cam

    if bbox:
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) >= 1000]
        # xtl, ytl, xbr, ybr = math.inf, math.inf, -math.inf, -math.inf
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(cam, (x, y), (x + w, y + h), (36, 255, 12), 2)
            # xtl, ytl = min(xtl, x), min(ytl, y)
            # xbr, ybr = max(xbr, x + w), max(ybr, y + h)
        # cv2.rectangle(cam, (xtl, ytl), (xbr, ybr), (36, 255, 12), 2)
    return cam


def main():
    torch.set_grad_enabled(False)

    parser = ArgParser()
    args = parser.parse_arguments()

    if args.dataset == "ave":
        vis_dataset = AVEDataset(args, phase="vis")
    else:
        vis_dataset = MusicDataset(args, phase="vis")

    meta_data = [(x[0], x[1]) for x in vis_dataset.data]
    str_to_evt_label = vis_dataset.str_to_evt_label

    evt_label_to_str = {}
    for k, v in str_to_evt_label.items():
        evt_label_to_str[v] = k

    dataloader = DataLoader(
        vis_dataset, batch_size=1, shuffle=False, num_workers=args.workers
    )

    # model
    model = STM(args)
    model = torch.nn.DataParallel(model)
    args.ckpt = os.path.join(args.ckpt, "{}.pth".format(args.kwckpt))
    ret = model.load_state_dict(torch.load(args.ckpt))
    model = model.module
    print("{} Sucessfully loaded checkpoint: {}".format(ret, args.ckpt))
    model.cuda()
    model.eval()

    tensor2img = tf.ToPILImage()
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    untf = tf.Compose([unorm, tensor2img])

    inference_dir = args.inference_dir
    inference_dir = os.path.join(inference_dir, args.id)
    # inference_dir = os.path.join(inference_dir, args.id, args.kwckpt)
    print("INFERENCE DIR: {}".format(inference_dir))

    if args.overwrite:
        if os.path.exists(inference_dir):
            shutil.rmtree(inference_dir)

    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    assert len(meta_data) == len(dataloader)
    for (i, data) in enumerate(tqdm(dataloader)):
        aud_file, vid_dir = meta_data[i]

        frame_count = len(os.listdir(vid_dir))
        orig_fps = math.ceil(frame_count / time_length)

        write_name = aud_file.split("/")[-1][:-4]

        if write_name[0] == "-":
            write_name = list(write_name)
            write_name[0] = "+"
            write_name = "".join(write_name)

        write_name = os.path.join(inference_dir, "{}.mp4".format(write_name))
        if not args.overwrite and os.path.exists(write_name):
            continue

        audios = data["audios"].cuda()
        frames = data["frames"].cuda()

        _, T, _, H, W = frames.size()

        f_o, p_o = model.downstream(audios, frames)
        # pred = model.classifier(f_o)

        video = []
        for t in range(T):
            image = untf(frames[-1, t].cpu())
            heatmap = p_o[-1, t].cpu().numpy()

            image = apply_heatmap(image, heatmap, (H, W), bbox=True)
            # draw gt box
            # xtl, ytl, xbr, ybr = boxes[t].cpu().numpy()
            # cv2.rectangle(image, (xtl, ytl), (xbr, ybr), (0, 0, 0), 2)

            # draw class
            # pred_cls = pred[-1, t].argmax().item()
            # pred_cls = evt_label_to_str[pred_cls]
            # image = cv2.putText(
            #     image, pred_cls, org, font, fontScale, color, thickness, cv2.LINE_AA
            # )

            video.append(image)

        audioclip = AudioFileClip(aud_file)
        videoclip = ImageSequenceClip(video, fps=int(orig_fps))
        videoclip = videoclip.set_audio(audioclip)
        videoclip.write_videofile(write_name)


if __name__ == "__main__":
    main()
