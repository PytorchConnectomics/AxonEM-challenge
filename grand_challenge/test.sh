#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
# ./build.sh

MEM_LIMIT="4g"


for i in "axonem_mouse axonH_gt_16nm.h5" "axonem_human axonH_gt_16nm.h5"; do
    set -- $i # convert the "tuple" into the param args $1 $2...
    VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)

    docker volume create axonem-output-$VOLUME_SUFFIX

    mkdir -p test
    rm -rf test/*
    cp ../test/$2 test/test-input.h5

        # Do not change any of the parameters to docker run, these are fixed
    docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v ${PWD}/test/:/input/ \
        -v axonem-output-$VOLUME_SUFFIX:/output/ \
        $1

    docker run --rm \
        -v axonem-output-$VOLUME_SUFFIX:/output/ \
        python:3.9-slim cat /output/metrics.json | python -m json.tool

    docker volume rm axonem-output-$VOLUME_SUFFIX
done
