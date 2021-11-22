import os
import glob
import argparse
from natsort import natsorted
import multiprocess.dump_videos as dump

parser = argparse.ArgumentParser()
parser.add_argument("--rank", default=None, type=int)
parser.add_argument("--world_size", default=None, type=int)
parser.add_argument("--root_dir", default=None, type=str)
args = parser.parse_args()
rank = args.rank
world_size = args.world_size

write_dir = f"{args.root_dir}/video/"
raw_dir = f"{args.root_dir}/raw_video/"
videos = glob.glob(os.path.join(raw_dir, "*.mp4"))
videos = natsorted(videos)

total_num_videos = len(videos)
print("[NODE RANK] {} - TOTAL VIDS {}".format(rank, total_num_videos))

dump.VIDEOS = videos
dump.WRITE_DIR = write_dir

videos_per_node = total_num_videos / world_size
node_start = int(rank * videos_per_node)
node_end = int((rank + 1) * videos_per_node - 1)

dump.multiprocess_videos_to_images(node_start, node_end)
