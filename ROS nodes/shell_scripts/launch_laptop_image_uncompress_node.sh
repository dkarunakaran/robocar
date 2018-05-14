#!/bin/bash

source activate image_processing_node
export PYTHONPATH="/home/dhanoop/anaconda3/envs/image_processing_node/lib/python3.4/site-packages:$PYTHONPATH"

rosrun image_transport republish compressed in:=/raspicam_node/image raw out:=/raspicam_node/image
