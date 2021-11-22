import os
import glob
import argparse
from natsort import natsorted
import multiprocess.dump_audios as dump

parser = argparse.ArgumentParser()
parser.add_argument("--rank", default=None, type=int)
parser.add_argument("--world_size", default=None, type=int)
parser.add_argument("--root_dir", default=None, type=str)
args = parser.parse_args()
rank = args.rank
world_size = args.world_size

write_dir = f"{args.root_dir}/audios/"
raw_dir = f"{args.root_dir}/raw_videos/"
videos = glob.glob(os.path.join(raw_dir, "*.mp4"))
videos = natsorted(videos)

if not os.path.exists(write_dir):
    os.makedirs(write_dir)

total_num_videos = len(videos)
print("[NODE RANK] {} - TOTAL VIDS {}".format(rank, total_num_videos))

dump.VIDEOS = videos
dump.WRITE_DIR = write_dir

videos_per_node = total_num_videos / world_size
node_start = int(rank * videos_per_node)
node_end = int((rank + 1) * videos_per_node - 1)

dump.multiprocess_videos_to_audios(node_start, node_end)
