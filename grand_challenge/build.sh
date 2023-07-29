#!/usr/bin/env bash

cd ..
docker build -t axonem_mouse -f ./grand_challenge/Dockerfile . --build-arg GROUND_TRUTH_PATH=./ground-truth/gt_mouse_16nm_skel_stats_gc.p
docker build -t axonem_human -f ./grand_challenge/Dockerfile . --build-arg GROUND_TRUTH_PATH=./ground-truth/gt_human_16nm_skel_stats_gc.p
