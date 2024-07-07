#!/bin/bash

evalset=(
    # standard-slow
    #standard-fast
    # slightly-dark-slow
    # slightly-dark-fast
    dark-slow
    # dark-fast
    # p006
    # deer_run
)

for seq in ${evalset[@]}; do
    # python evaluation_scripts/test_tum.py --datapath=$TUM_PATH/$seq --weights=droid.pth --disable_vis $@
    python evaluate.py --model=checkpoints/setting3-2D/things.pth --dataset=flything3D --mixed_precision --split=$seq  $@

done

