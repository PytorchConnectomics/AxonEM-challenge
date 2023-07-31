#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
TESTDIR="$(dirname "$SCRIPTPATH")/test/"
./build.sh


MEM_LIMIT="4g"

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)

docker volume create axonem-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
    --memory="${MEM_LIMIT}" \
    --memory-swap="${MEM_LIMIT}" \
    --network="none" \
    --cap-drop="ALL" \
    --security-opt="no-new-privileges" \
    --shm-size="128m" \
    --pids-limit="256" \
    -v ${TESTDIR}:/input/ \
    -v axonem-output-$VOLUME_SUFFIX:/output/ \
    axonem

docker run --rm \
    -v axonem-output-$VOLUME_SUFFIX:/output/ \
    python:3.9-slim cat /output/metrics.json | python -m json.tool

docker volume rm axonem-output-$VOLUME_SUFFIX
