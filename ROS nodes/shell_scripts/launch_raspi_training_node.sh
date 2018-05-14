#!/bin/bash
export PATH=/home/dhanoop/miniconda3/bin:$PATH
source activate training_node
export PYTHONPATH="/home/dhanoop/miniconda3/envs/training_node/lib/python3.4/site-packages:$PYTHONPATH"
cd /home/dhanoop/catkin_ws/src/training_node/scripts
rosrun training_node training_node_raspi.py
