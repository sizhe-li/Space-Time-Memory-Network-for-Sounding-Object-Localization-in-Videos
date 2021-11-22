import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import find_bbox_in_heatmap, ModelBuilder
from . import calc_iou, calc_ciou, accuracy


class Encoder(nn.Module):
    def __init__(self, a_net, v_net, train_vision):
        super(Encoder, self).__init__()

        if not train_vision:
            # same as context-dependent torch.no_grad()
            for param in v_net.parameters():
                param.requires_grad = False
        self.a_net = a_net
        self.v_net = v_net

    def forward(self, audio, frame):
        f_a = self.a_net(audio)
        f_v = self.v_net(frame)

        return f_a, f_v


class Match(nn.Module):
    def __init__(self, indim_a, indim_v, outdim, normalize):
        super(Match, self).__init__()

        self.a_proj = nn.Sequential(
            nn.Conv2d(indim_a, indim_a // 2, 1),
            nn.BatchNorm2d(indim_a // 2),
            nn.ReLU(),
            nn.Conv2d(indim_a // 2, outdim, 1),
            nn.BatchNorm2d(outdim),
        )

        self.v_proj = nn.Sequential(
            nn.Conv2d(indim_v, indim_v // 2, 1),
            nn.BatchNorm2d(indim_v // 2),
            nn.ReLU(),
            nn.Conv2d(indim_v // 2, indim_v // 4, 1),
            nn.BatchNorm2d(indim_v // 4),
            nn.ReLU(),
            nn.Conv2d(indim_v // 4, outdim, 1),
            nn.BatchNorm2d(outdim),
        )
        self.normalize = normalize
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, f_a, f_v):
        f_a = self.a_proj(f_a)  # B, C, H, W
        f_v = self.v_proj(f_v)  # B, C, H, W

        f_a = self.pool(f_a)
        f_a = f_a.view(*f_a.shape[:2])  # B, C

        if self.normalize:
            f_a = F.normalize(f_a, p=2, dim=1)
            f_v = F.normalize(f_v, p=2, dim=1)

        return f_a, f_v


class Memory(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        MultiheadAttention
        """
        super(Memory, self).__init__()
        self.attn_layer = nn.MultiheadAttention(embed_dim, num_heads)
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, q, m):
        B, T, C, H, W = m.size()

        # query
        # in: B, C, H, W
        # want: 1, B, C
        q = self.pool(q)
        q = q.view(1, B, C)

        # memory
        # in: B, T, C, H, W
        # want: T, B, C
        mem = m.contiguous().view(B * T, C, H, W)
        mem = self.pool(mem)
        mem = mem.view(B, T, C).permute(1, 0, 2)  # T, B, C

        # out: [1, B, C]; [B, 1, T]
        _, weights = self.attn_layer(q, mem, mem)
        weights = weights.view(B, T, 1, 1, 1)

        m = (weights * m).sum(1)  #
        return m


class STM(nn.Module):
    def __init__(self, args, outdim=256):
        super(STM, self).__init__()
        self.use_mem_o = args.use_mem_o
        self.baseline = args.baseline
        self.img_size = args.img_size
        self.T = int(args.T / 2)
        # encode
        builder = ModelBuilder()
        a_net = builder.build_sound(
            arch=args.arch_sound, fc_dim=args.num_channels, weights=args.weights_sound
        )
        v_net = builder.build_video(arch=args.arch_video, weights=args.weights_video)
        self.encoder = Encoder(a_net, v_net, args.train_vision)
        # match
        self.match = Match(
            indim_a=a_net.get_embed_dim(),
            indim_v=v_net.get_embed_dim(),
            outdim=outdim,
            normalize=args.normalize,
        )
        if not self.baseline:
            # memory
            self.memory_a = Memory(a_net.get_embed_dim(), args.num_heads)
            self.memory_v = Memory(v_net.get_embed_dim(), args.num_heads)
            self.memory_o = Memory(outdim, args.num_heads)
            self.downstream = self.downstream_stm
        else:
            self.downstream = self.downstream_baseline

        if args.cls_pool == "max":
            self.cls_pool = lambda x: torch.max(x, dim=1)[0]
        elif args.cls_pool == "mean":
            self.cls_pool = lambda x: torch.mean(x, dim=1)
        else:
            raise NotImplementedError
        # criterion
        self.crit = nn.CrossEntropyLoss()
        self.accuracy = lambda x, y: accuracy(x, y)[0].item()
        if args.dataset == "ave":
            # classifier
            self.classifier = nn.Sequential(nn.Linear(outdim * 2, outdim),
                                            nn.ReLU(),
                                            nn.Linear(outdim, 28))
        else:
            self.classifier = nn.Linear(outdim * 2, 13)

    def create_optimizer(self, args):
        param_groups = [
            # encoder
            {"params": self.encoder.a_net.parameters(), "lr": args.lr_sound},
            {"params": self.encoder.v_net.parameters(), "lr": args.lr_video},
            # match
            {"params": self.match.parameters(), "lr": args.lr_aggre},
            # classifier
            {"params": self.classifier.parameters(), "lr": args.lr_cls},
        ]
        if not self.baseline:
            print("Adding memory params to optimizer!", flush=True)
            param_groups.extend(
                [
                    # memory
                    {"params": self.memory_a.parameters(), "lr": args.lr_aggre},
                    {"params": self.memory_v.parameters(), "lr": args.lr_aggre},
                ]
            )
            if self.use_mem_o:
                param_groups.extend(
                    [
                        {"params": self.memory_o.parameters(), "lr": args.lr_aggre},
                    ]
                )

        return torch.optim.Adam(param_groups)

    def forward(self, data, gpu):
        # B, T, 1, H, W
        audios = data["audios"].cuda(gpu)
        frames = data["frames"].cuda(gpu)
        labels = data["labels"].cuda(gpu)

        B, T = frames.shape[:2]

        # pos
        f_o, _ = self.downstream(audios, frames)  # B, T, C
        pred = self.classifier(f_o)  # B, T, Cls
        pred = self.cls_pool(pred)
        loss = self.crit(pred, labels)
        acc = self.accuracy(pred, labels)

        return loss, acc

    def A_attend_V(self, f_a, f_v):
        """
        Args:
            f_a : (B, C)
            f_v : (B, C, H, W)
        """

        B, C, H, W = f_v.size()

        fa = f_a.view(B, C, 1)
        fv = f_v.view(B, C, H * W)  # B, C, H*W
        fv_t = torch.transpose(fv, 1, 2)  # B, H*W, C
        p = torch.bmm(fv_t, fa)  # B, H*W, 1
        p = F.softmax(p, dim=1)
        p = p.view(B, H, W)

        return p

    def downstream_baseline(self, audios, frames):
        B, T = audios.shape[:2]
        # encode
        audios = audios.view(B * T, *audios.shape[2:])
        frames = frames.view(B * T, *frames.shape[2:])
        f_a, f_v = self.encoder(audios, frames)
        f_a, f_v = self.match(f_a, f_v)  # B*T, C;  B*T, C, H, W
        p = self.A_attend_V(f_a, f_v)  # B*T, H, W
        f_o = p.unsqueeze(1) * f_v
        f_o = f_o.sum((-2, -1))
        f_o = torch.cat([f_a, f_o], dim=-1)
        f_o = f_o.view(B, T, *f_o.shape[1:])
        p_o = p.view(B, T, *p.shape[1:])

        return f_o, p_o

    def downstream_stm(self, audios, frames):
        B, T = audios.shape[:2]
        # encode
        audios = audios.view(B * T, *audios.shape[2:])
        frames = frames.view(B * T, *frames.shape[2:])
        f_a, f_v = self.encoder(audios, frames)

        f_o = None
        if self.use_mem_o:
            f_a_single, f_v_single = self.match(f_a, f_v)
            p = self.A_attend_V(f_a_single, f_v_single)
            f_o = p.unsqueeze(1) * f_v_single
            f_o = f_o.view(B, T, *f_o.shape[1:])

        f_a = f_a.view(B, T, *f_a.shape[1:])
        f_v = f_v.view(B, T, *f_v.shape[1:])

        f_mem_o = []
        p_mem_o = []
        for t in range(T):
            mem_a = f_a[:, t - self.T: t + 1 + self.T]
            mem_v = f_v[:, t - self.T: t + 1 + self.T]

            f_a_t, f_v_t = f_a[:, t], f_v[:, t]
            f_a_t = self.memory_a(f_a_t, mem_a)
            f_v_t = self.memory_v(f_v_t, mem_v)
            f_a_t, f_v_t = self.match(f_a_t, f_v_t)

            p = self.A_attend_V(f_a_t, f_v_t)  # B, H, W
            f_o_t = p.unsqueeze(1) * f_v_t

            if self.use_mem_o:
                mem_o = f_o[:, t - self.T: t + 1 + self.T]
                f_o_t = self.memory_o(f_o_t, mem_o)
                p = self.A_attend_V(f_a_t, f_o_t)
            f_o_t = f_o_t.sum((-2, -1))
            f_o_t = torch.cat([f_a_t, f_o_t], dim=-1)
            f_mem_o.append(f_o_t)
            p_mem_o.append(p)

        f_mem_o = torch.stack(f_mem_o, dim=1)  # B, T, C
        p_mem_o = torch.stack(p_mem_o, dim=1)  # B, T, H, W

        return f_mem_o, p_mem_o
