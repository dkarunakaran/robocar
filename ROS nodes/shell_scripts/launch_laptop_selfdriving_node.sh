#!/bin/bash
export PATH=/home/dhanoop/anaconda3/bin:$PATH
unset PYTHONPATH
source activate image_processing_node
export PYTHONPATH="/home/dhanoop/anaconda3/envs/image_processing_node/lib/python3.5/site-packages:/opt/ros/kinetic/lib/python2.7/dist-packages:/usr/lib/python2.7/site-packages"

cd /home/dhanoop/catkin_ws/src/image_processing_node/scripts
rosrun image_processing_node image_processing_node.py
