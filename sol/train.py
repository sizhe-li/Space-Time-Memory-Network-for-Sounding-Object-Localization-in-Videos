import os
import time
import random
import pprint

# third party
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter

from sol.dataset import AVEDataset, MusicDataset
from sol.models.stm_nets import STM
from sol.arguments import ArgParser
from sol.utils import makedirs, setup_for_distributed, MetricLogger, save_on_master


def train(model, dataloader, optimizer, epoch, args):
    print("Training at epoch {}".format(epoch), flush=True)
    torch.set_grad_enabled(True)
    model.train()

    logger = MetricLogger(delimiter="  ")
    print_freq = args.disp_iter
    header = "Train:"
    tic = time.perf_counter()
    for data in logger.log_every(dataloader, print_freq, header):
        logger.update(data_time=time.perf_counter() - tic)
        model.zero_grad()

        loss, acc = model(data, args.gpu)
        logger.update(loss=loss.item())
        logger.update(accuracy=acc)
        loss.backward()
        optimizer.step()

        tic = time.perf_counter()

    logger.synchronize_between_processes()

    if not args.distributed or (args.distributed and args.is_master):
        writer = SummaryWriter(log_dir=args.vis_dir)
        msg = "[Train] *"
        avg_loss = logger.meters["loss"].global_avg
        avg_acc = logger.meters["accuracy"].global_avg
        msg += " loss: {:.4f}".format(avg_loss)
        msg += " acc@1: {:.4f}".format(avg_acc)
        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("acc/train", avg_acc, epoch)
        writer.close()
        print(msg, flush=True)


def checkpoint(model, epoch, args):
    print("[Epoch {}] saving checkpoint...\n".format(epoch), flush=True)

    save_on_master(model.state_dict(), "{}/latest.pth".format(args.ckpt))
    save_on_master(model.state_dict(), "{}/epoch{}.pth".format(args.ckpt, epoch))
    if args.cur_benchmark >= args.best_benchmark:
        args.best_benchmark = args.cur_benchmark
        save_on_master(model.state_dict(), "{}/best.pth".format(args.ckpt))


def main(gpu, args):
    args.gpu = gpu
    cudnn.benchmark = True
    if args.distributed:
        global_rank = args.rank * args.ngpus_per_node + args.gpu
        args.is_master = global_rank == 0
        dist.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=global_rank,
        )
        # only print on master process
        setup_for_distributed(args.is_master)

    model = STM(args)

    optimizer = model.create_optimizer(args)
    if args.distributed:
        args.batch_size = int(args.batch_size / args.world_size)
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
    else:
        model.cuda()

    if args.dataset == "ave":
        dataset_train = AVEDataset(args, phase="train")
    else:
        dataset_train = MusicDataset(args, phase="train")

    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(
            dataset_train, num_replicas=args.world_size, rank=global_rank
        )
    else:
        sampler_train = None

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=(sampler_train is None),
        sampler=sampler_train,
        pin_memory=False,
        num_workers=int(args.workers),
        drop_last=True,
    )

    # training
    for epoch in range(1, args.num_epochs + 1):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        print("*=*=*=*=*=*=*=*=Epoch {}*=*=*=*=*=*=*=*=".format(epoch), flush=True)
        train(model, loader_train, optimizer, epoch, args)

        if epoch % args.eval_freq == 0:
            if not args.distributed or (args.distributed and args.is_master):
                checkpoint(model, epoch, args)

    print("Training Complete!", flush=True)


if __name__ == "__main__":
    parser = ArgParser()
    args = parser.parse_arguments()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not args.distributed or (args.distributed and args.rank == 0):
        parser.print_arguments(args)
        print("Model ID: {}".format(args.id))
        if args.mode == "train":
            makedirs(args.vis_dir, remove=True)
            makedirs(args.ckpt, remove=True)

    args.best_benchmark = 0  # miou
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node * args.nodes
        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = args.master_port
        mp.spawn(main, nprocs=args.ngpus_per_node, args=(args,))

    else:
        main(None, args)
