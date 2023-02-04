#!/bin/bash
CUDA_VISIBLE_DEVICES=0 nohup python3 -u Source/main.py \
                        --mode train \
                        --batchSize 1 \
                        --gpu 1 \
                        --trainListPath ./Datasets/Stereo/scene_flow_debug_training_list.csv \
                        --imgWidth 512 \
                        --imgHeight 256 \
                        --dataloaderNum 4 \
                        --maxEpochs 45 \
                        --imgNum 35454 \
                        --sampleNum 1 \
                        --lr 0.001 \
                        --dist False \
                        --modelName GwcNet \
                        --outputDir ./DebugResult/ \
                        --modelDir ./DebugCheckpoint/ \
                        --dataset sceneflow > TrainRun.log 2>&1 &

tail -f  TrainRun.log