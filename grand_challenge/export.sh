#!/usr/bin/env bash

./build.sh

docker save axonem_mouse | gzip -c > axonem_mouse.tar.gz
docker save axonem_human | gzip -c > axonem_human.tar.gz
