#!/bin/bash

python main.py --input_picture_path /home/gxp/Projects/dfl_sandbox/mv/workspace/data_src/aligned --output_picture_path /home/gxp/Projects/dfl_sandbox/mv/workspace/data_src/aligned_cluster --min_sim 0.2 --k 600

python main.py --input_picture_path /home/gxp/Projects/dfl_sandbox/mv/workspace/data_dst/aligned --output_picture_path /home/gxp/Projects/dfl_sandbox/mv/workspace/data_dst/aligned_cluster --min_sim 0.2 --k 600
