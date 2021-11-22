#!/bin/bash

echo "rank=$1"
python3 extract_frames.py --rank $1 --world_size 3 --root_dir /u/sli96/data/music15set/data/train
