import os
import cv2
import math
import glob
import torch
import torchaudio
import torchvision
import random
import pickle
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F

from PIL import Image
from . import transforms as vtransforms
from . import mel_features
from . import vggish_params
from torchvision import transforms as tf
from torch.utils.data import DataLoader, Dataset
from natsort import natsorted


def _read_annotations_xml(annot_file):
    tree = ET.parse(annot_file)
    root = tree.getroot()

    orig_size = root.find("meta").find("task").find("original_size")
    H, W = orig_size.find("height").text, orig_size.find("width").text
    orig_size = (int(H), int(W))
    labels = {}

    for track in root.iter("track"):
        for box in track.iter("box"):
            if not int(box.get("outside")):
                frame_index = int(box.get("frame"))
                xtl, ytl, xbr, ybr = (
                    box.get("xtl"),
                    box.get("ytl"),
                    box.get("xbr"),
                    box.get("ybr"),
                )
                coord = (float(xtl), float(ytl), float(xbr), float(ybr))
                labels[frame_index] = labels.get(frame_index, []) + [coord]

    return orig_size, labels


class AVEDataset(Dataset):
    def __init__(self, args, phase):

        # preparing data
        data_dir = os.path.join(args.root, "AVE_Dataset")
        video_dir = os.path.join(data_dir, "videos")
        audio_dir = os.path.join(data_dir, "audios")

        config = os.path.join(data_dir, "trainSet.txt")
        self._sample_data = self._sample_train

        with open(config, "r") as f:
            sample_list = f.readlines()
        sample_list = [x.strip("\n").split("&") for x in sample_list]

        data = []
        for (evt, name, _, _, _) in sample_list:
            aud_file = os.path.join(audio_dir, name + ".wav")
            vid_dir = os.path.join(video_dir, name)
            if os.path.exists(aud_file) and os.path.isdir(vid_dir):
                data.append((aud_file, vid_dir, evt))

        self.data = data

        # params
        self.time_length = args.time_length
        self.frame_rate = args.frame_rate
        self.img_size = args.img_size
        self.phase = phase
        self.N = math.ceil(args.T / args.frame_rate)
        self.T = args.T
        # video transform
        self._init_vtransform()
        self.str_to_evt_label = {
            "Accordion": 0,
            "Acoustic guitar": 1,
            "Baby cry, infant cry": 2,
            "Banjo": 3,
            "Bark": 4,
            "Bus": 5,
            "Cat": 6,
            "Chainsaw": 7,
            "Church bell": 8,
            "Clock": 9,
            "Female speech, woman speaking": 10,
            "Fixed-wing aircraft, airplane": 11,
            "Flute": 12,
            "Frying (food)": 13,
            "Goat": 14,
            "Helicopter": 15,
            "Horse": 16,
            "Male speech, man speaking": 17,
            "Mandolin": 18,
            "Motorcycle": 19,
            "Race car, auto racing": 20,
            "Rodents, rats, mice": 21,
            "Shofar": 22,
            "Toilet flush": 23,
            "Train horn": 24,
            "Truck": 25,
            "Ukulele": 26,
            "Violin, fiddle": 27,
            "background": 28,
        }

    def get_label_str(self, label):
        for event, idx in self.str_to_evt_label.items():
            if label == idx:
                return event

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._sample_data(idx)

    def _get_labels(self, evt):
        target = self.str_to_evt_label[evt]
        return torch.tensor(target)

    def _sample_train(self, idx):

        aud_file, vid_dir, evt = self.data[idx]

        t = random.choice(range(self.time_length))
        sampl_range = list(range(self.time_length))
        if len(range_1 := sampl_range[t-(self.N-1):t+1]) > len(range_2 := sampl_range[t:t+self.N]):
            sampl_range = range_1
        else:
            sampl_range = range_2
        audios = []
        frames = []
        for t in sampl_range:
            audios.append(self._load_audios_given_sec(aud_file, t))
            frames.append(self._load_frames_given_sec(vid_dir, t))
        audios = torch.cat(audios, 0)[:self.T]
        frames = torch.cat(frames, 0)[:self.T]
        labels = self._get_labels(evt)

        ret_dict = {
            "audios": audios,
            "frames": frames,
            "labels": labels,
        }

        return ret_dict

    def _sample_val(self, idx):
        aud_file, vid_dir, orig_size, annot_labels, evt = self.data[idx]

        # load av
        audios_annot, frames_annot = self._load_av_annot(
            aud_file, vid_dir, annot_labels.keys()
        )

        # load boxes
        orig_h, orig_w = orig_size
        x_ratio = self.img_size / orig_w
        y_ratio = self.img_size / orig_h
        boxes = []
        gtmap = []
        for _, coords in annot_labels.items():

            # load gt box
            xtl, ytl, xbr, ybr = math.inf, math.inf, -math.inf, -math.inf
            for (x1, y1, x2, y2) in coords:
                # merging boxes
                xtl, ytl = min(xtl, x1), min(ytl, y1)
                xbr, ybr = max(xbr, x2), max(ybr, y2)
            xtl *= x_ratio
            xbr *= x_ratio
            ytl *= y_ratio
            ybr *= y_ratio
            bbox = torch.tensor([xtl, ytl, xbr, ybr]).float()
            _map = torch.zeros((self.img_size, self.img_size))
            _map[int(ytl) : int(ybr), int(xtl) : int(xbr)] = 1.0
            boxes.append(bbox)
            gtmap.append(_map)

        # T, C, H, W
        labels = self._get_labels(evt)
        boxes = torch.stack(boxes, 0)
        gtmap = torch.stack(gtmap, 0)

        ret_dict = {
            "audios": audios_annot,
            "frames": frames_annot,
            "labels": labels,
            "boxes": boxes,
            "gtmap": gtmap,
        }

        return ret_dict

    def _sample_vis(self, idx):
        aud_file, vid_dir, orig_size, annot_labels, evt = self.data[idx]

        audios, frames = self._load_av_annot(aud_file, vid_dir, frame_nums=None)

        ret_dict = {"audios": audios, "frames": frames}

        return ret_dict

    def _load_frames_given_sec(self, video_dir, time_index):
        frame_paths = natsorted(os.listdir(video_dir))
        frame_count = len(frame_paths)
        frame_interval = int(frame_count / self.time_length)
        frame_nums = self._sample_frames(frame_interval, time_index)
        frames = [
            Image.open(os.path.join(video_dir, frame_paths[num])) for num in frame_nums
        ]

        return frames

    def _load_audios_given_sec(self, audio_file, time_index):
        audio_raw, rate = self._load_raw_audios(audio_file)
        # cutting into segments
        mul_1 = time_index * self.frame_rate
        mul_2 = (1 / self.frame_rate) * rate
        audio_segments = []
        for seg_index in range(self.frame_rate):
            seg = self._stft_audio_seg_index(audio_raw, mul_1, mul_2, seg_index)
            audio_segments.append(seg)
        audio_segments = torch.stack(audio_segments)  # T, 1, H, W

        return audio_segments

    def _load_av_annot(self, audio_file, video_dir, frame_nums):
        audio_raw, rate = self._load_raw_audios(audio_file)
        frame_paths = natsorted(os.listdir(video_dir))
        frame_count = len(frame_paths)
        if frame_nums is None:
            frame_nums = range(frame_count)

        video_annot = []
        audio_annot = []
        for num in frame_nums:
            # load audio
            mul_1, mul_2, seg_index = self.frame_index_to_seg(frame_count, rate, num)
            seg = self._stft_audio_seg_index(audio_raw, mul_1, mul_2, seg_index)
            audio_annot.append(seg)

            img = Image.open(os.path.join(video_dir, frame_paths[num]))
            video_annot.append(img)

        audio_annot = torch.stack(audio_annot, dim=0)
        video_annot = self.vid_transform(video_annot)

        return audio_annot, video_annot

    def _load_raw_audios(self, audio_file):
        audio_raw, rate = torchaudio.load(audio_file)
        if audio_raw.size(1) > self.time_length * rate:
            audio_raw = audio_raw[:, : self.time_length * rate]
        # convert to mono sound
        if audio_raw.size(0) > 1:
            audio_raw = torch.mean(audio_raw, dim=0)
        audio_raw = audio_raw.view(-1)
        audio_raw = audio_raw.numpy().astype(np.float32)
        # repeat if audio is too short
        if audio_raw.shape[0] < rate * self.time_length:
            n = int(rate * self.time_length / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        return audio_raw, rate

    def _sample_frames(self, frame_interval, time_index):
        num = []

        for i in range(self.frame_rate):
            num.append(
                int(
                    time_index * frame_interval
                    + (i * 1.0 / self.frame_rate) * frame_interval
                )
            )
        return num

    def _stft_audio_seg_index(self, audio_raw, mul_1, mul_2, seg_index):
        start = int((mul_1 + seg_index) * mul_2)
        end = int((mul_1 + seg_index + 1) * mul_2)
        return self._perform_stft(audio_raw[start:end])

    def _perform_stft(self, audio):
        # STFT
        ampN = self.waveform_to_examples(audio)
        mags = torch.from_numpy(ampN).type(torch.FloatTensor).unsqueeze(0)

        return mags

    def waveform_to_examples(self, data):
        """Converts audio waveform into an array of examples for VGGish.

        Args:
          data: np.array of either one dimension (mono) or two dimensions
            (multi-channel, with the outer dimension representing channels).
            Each sample is generally expected to lie in the range [-1.0, +1.0],
            although this is not required.
          sample_rate: Sample rate of data.

        Returns:
          3-D np.array of shape [num_examples, num_frames, num_bands] which represents
          a sequence of examples, each of which contains a patch of log mel
          spectrogram, covering num_frames frames of audio and num_bands mel frequency
          bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
        """

        # Compute log mel spectrogram features.
        log_mel = mel_features.log_mel_spectrogram(
            data,
            audio_sample_rate=vggish_params.SAMPLE_RATE,
            log_offset=vggish_params.LOG_OFFSET,
            window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
            hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
            num_mel_bins=vggish_params.NUM_MEL_BINS,
            lower_edge_hertz=vggish_params.MEL_MIN_HZ,
            upper_edge_hertz=vggish_params.MEL_MAX_HZ,
        )

        return log_mel

    def frame_index_to_seg(self, frame_count, rate, index):
        fps = math.ceil(frame_count / self.time_length)

        frame_per_segment = math.ceil(fps / self.frame_rate)

        time_index = index // fps
        seg_index = index % fps  # get remainder
        seg_index = seg_index // frame_per_segment
        mul_1 = time_index * self.frame_rate
        mul_2 = (1 / self.frame_rate) * rate
        return mul_1, mul_2, seg_index

    # video transform funcs
    def _init_vtransform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.phase == "train":
            transform_list.append(
                vtransforms.Resize(int(self.img_size * 1.1), Image.BICUBIC)
            )
            transform_list.append(vtransforms.RandomCrop(self.img_size))
            transform_list.append(vtransforms.RandomHorizontalFlip())
        else:
            transform_list.append(
                vtransforms.Resize((self.img_size, self.img_size), Image.BICUBIC)
            )
            # transform_list.append(vtransforms.CenterCrop(self.img_size))

        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
        transform_list.append(vtransforms.Stack())
        self.vid_transform = tf.Compose(transform_list)


class MusicDataset(AVEDataset):
    def __init__(self, args, phase="train"):
        data_dir = os.path.join(args.root, "music15set", "data")

        data_sheet = pd.read_csv(os.path.join(data_dir, "train_sep_new.csv"))
        data_dir = os.path.join(data_dir, "train")
        self._sample_data = self._sample_train

        video_dir = os.path.join(data_dir, "video")
        audio_dir = os.path.join(data_dir, "audio")

        data = []
        for _, row in data_sheet.iterrows():
            keyword = row["keyword"]
            evt = row["target"]
            aud_file = os.path.join(audio_dir, keyword + ".wav")
            vid_dir = os.path.join(video_dir, keyword)
            if os.path.exists(aud_file) and os.path.isdir(vid_dir):
                data.append((aud_file, vid_dir, evt))
        self.data = data

        # params
        self.time_length = args.time_length
        self.frame_rate = args.frame_rate
        self.img_size = args.img_size
        self.phase = phase
        self.N = math.ceil(args.T / args.frame_rate)
        self.T = args.T
        # video transform
        self._init_vtransform()
        self.str_to_evt_label = {
            "accordion": 0,
            "guitar": 1,
            "banjo": 2,
            "cello": 3,
            "drum": 4,
            "flute": 5,
            "harmonica": 6,
            "harp": 7,
            "marimba": 8,
            "piano": 9,
            "saxophone": 10,
            "french horn": 11,
            "trombone": 11,
            "trumpet": 11,
            "violin": 12,
        }

    def _get_labels(self, evt):

        evt = evt.split(", ")
        if len(evt) > 1:
            if "marimba" in evt:
                evt = "marimba"
            elif "violin" in evt:
                evt = "violin"
            else:
                evt = random.choice(evt)
        else:
            evt = evt[0]
        target = self.str_to_evt_label[evt]

        return torch.tensor(target)
