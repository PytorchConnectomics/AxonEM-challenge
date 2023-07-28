#!/usr/bin/env bash

./build.sh

docker save axonem | gzip -c > axonem.tar.gz
