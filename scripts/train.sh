#!/bin/bash
set -e
while echo $1 | grep -q ^-; do
    eval $( echo $1 | sed 's/^-//' )=$2
    shift
    shift
done

T=$T
root=$root
dataset=$dataset
workers=$workers
num_epochs=$num_epochs

MASTER=$(/bin/hostname -s)
MPORT=$(shuf -i8000-9999 -n1)


PYTHON_COMMANDS="--version stm_rebuttal \
                --mode train \
                --root $root \
                --dataset $dataset \
                --img_size 256 \
                --eval_freq 1 \
                --arch_sound vggish \
                --arch_video resnet152 \
                --T $T \
                --num_heads 1 \
                --cls_pool max \
                --frame_rate 4 \
                --batch_size 128 \
                --num_epochs $num_epochs \
                --workers $workers \
                --master_addr $MASTER \
                --master_port $MPORT \
                --nodes 1 \
                --rank 0 \
                --distributed \
                --normalize"

python3 -m sol.main $PYTHON_COMMANDS
