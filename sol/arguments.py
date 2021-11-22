import os
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--dataset",
            choices=["ave", "music"],
            required=True,
            type=str,
            help="which dataset",
        )
        # Model related arguments
        parser.add_argument(
            "--arch_sound", default="vggish", help="architecture of net_sound"
        )
        parser.add_argument(
            "--arch_video", default="resnet50", help="architecture of net_frame"
        )
        parser.add_argument(
            "--arch_match", default="base", help="architecture of net_match"
        )
        parser.add_argument(
            "--num_channels",
            default=32,
            type=int,
            help="number of channels for audio net",
        )
        parser.add_argument(
            "--weights_sound", default="", help="weights to finetune net_sound"
        )
        parser.add_argument(
            "--weights_video", default="", help="weights to finetune net_video"
        )
        parser.add_argument(
            "--weights_match", default="", help="weights to finetune net_synthesizer"
        )
        parser.add_argument(
            "--T",
            default=4,
            type=int,
            help="num attention heads for memory module"
        )
        parser.add_argument(
            "--num_heads",
            default=1,
            type=int,
            help="num attention heads for memory module",
        )
        parser.add_argument(
            "--cls_pool",
            default="max",
            choices=["max", "min", "mean"],
            type=str,
            help="pool type for classifier",
        )
        # Data related arguments
        parser.add_argument(
            "--num_gpus", default=1, type=int, help="number of gpus per node"
        )
        parser.add_argument(
            "--batch_size", default=32, type=int, help="input batch size"
        )
        parser.add_argument(
            "--workers",
            default=16,
            type=int,
            help="number of dataloder workers per process",
        )
        parser.add_argument(
            "--num_val", default=-1, type=int, help="number of images to evalutate"
        )

        parser.add_argument(
            "--time_length", default=10, type=int, help="time length (secs)"
        )
        parser.add_argument(
            "--img_size", default=256, type=int, help="size of input frame"
        )
        parser.add_argument(
            "--frame_rate", default=4, type=int, help="video frame sampling rate"
        )
        parser.add_argument(
            "--num_boxes", default=10, type=int, help="number of boxes to extract"
        )
        # Misc arguments
        parser.add_argument("--seed", default=54321, type=int, help="manual seed")
        parser.add_argument(
            "--disp_iter", type=int, default=5, help="frequency to display"
        )
        parser.add_argument(
            "--eval_freq", type=int, default=1, help="frequency to evaluate"
        )
        parser.add_argument(
            "--vis_freq", type=int, default=5, help="frequency to visualize"
        )
        parser.add_argument("--vis_epoch", type=int, default=10)
        # Distributed Training Parameters
        parser.add_argument("--distributed", action="store_true")
        parser.add_argument("--nodes", default=1, type=int)
        parser.add_argument(
            "--rank", default=0, type=int, help="ranking within the nodes"
        )
        parser.add_argument("--master_addr", default=None, type=str)
        parser.add_argument("--master_port", default=None, type=str)
        parser.add_argument(
            "--dist_url",
            default="env://",
            help="url used to set up distributed training",
        )
        self.parser = parser

    def add_train_arguments(self):
        parser = self.parser
        parser.add_argument("--version", default=None, type=str)
        parser.add_argument("--mode", default="train", help="[train/eval]")
        parser.add_argument(
            "--root", default="/media/lester/lester/", help="root directory"
        )
        # optimization related arguments
        parser.add_argument(
            "--num_epochs", default=100, type=int, help="epochs to train for"
        )
        parser.add_argument("--lr_sound", default=3e-4, type=float, help="LR")
        parser.add_argument("--lr_video", default=3e-4, type=float, help="LR")
        parser.add_argument("--lr_aggre", default=1e-3, type=float, help="LR")
        parser.add_argument("--lr_cls", default=1e-3, type=float, help="LR")
        parser.add_argument(
            "--lr_steps",
            nargs="+",
            type=int,
            default=[40, 60],
            help="steps to drop LR in epochs",
        )

        parser.add_argument("--train_vision", action="store_true")
        parser.add_argument("--normalize", action="store_true")
        parser.add_argument("--use_mem_o", action="store_true")
        parser.add_argument("--baseline", action="store_true")
        parser.add_argument(
            "--weight_decay", default=1e-4, type=float, help="weights regularizer"
        )

        self.parser = parser

    def add_inference_arguments(self):
        parser = self.parser
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="whether overwrite old visualizations",
        )
        parser.add_argument(
            "--kwckpt",
            type=str,
            default="best",
            help="keyword for checkpoint file during inference",
        )
        self.parser = parser

    def print_arguments(self, args):
        print("Input arguments:")
        for key, val in vars(args).items():
            print("{:16} {}".format(key, val))

    def parse_arguments(self):
        self.add_train_arguments()
        self.add_inference_arguments()
        args = self.parser.parse_args()
        if args.dataset == "ave":
            args.ckpt = os.path.join(args.root, "AVE_Dataset/experiments")
            args.inference_dir = os.path.join(args.root, "AVE_Dataset/inference")
        else:
            args.ckpt = os.path.join(args.root, "music15set/experiments")
            args.inference_dir = os.path.join(args.root, "music15set/inference")

        args.id = "-".join([args.version, args.dataset])
        args.id = "-".join([args.id, "baseline_{}".format(str(args.baseline).lower())])
        args.id = "-".join([args.id, args.arch_sound, args.arch_video])
        args.id = "-".join(
            [args.id, "num_heads_{}".format(str(args.num_heads).lower())]
        )
        args.id = "-".join([args.id, "cls_pool_{}".format(str(args.cls_pool).lower())])
        args.id = "-".join([args.id, "use_mem_o_{}".format(str(args.use_mem_o).lower())])
        args.id = "-".join(
            [args.id, "train_vision_{}".format(str(args.train_vision).lower())]
        )
        args.id = "-".join(
            [args.id, "normalize_{}".format(str(args.normalize).lower())]
        )
        args.id = "-".join(
            [args.id, "eval_freq_{}".format(str(args.eval_freq).lower())]
        )
        args.vis_dir = os.path.join(args.ckpt, "vis", args.id)
        args.ckpt = os.path.join(args.ckpt, args.id)

        return args
