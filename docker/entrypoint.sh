#!/bin/sh
echo args is $@
cd TAT
$1 setup/$2.py bdist_wheel -p manylinux2014_x86_64 -d /dist
