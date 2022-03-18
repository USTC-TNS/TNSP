#!/bin/sh
echo usage example: docker run -it --rm -v $PWD:/TAT -v /TAT/build image_name python3.10 PyTAT
cd TAT
$1 setup/$2.py bdist_wheel -p manylinux2014_x86_64
