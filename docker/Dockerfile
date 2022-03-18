FROM quay.io/pypa/manylinux2014_x86_64:latest

RUN yum install lapack-devel --assumeyes

RUN git clone https://git.ustc.edu.cn/hzhangxyz/TAT.git /TAT --branch dev

RUN git clone https://git.ustc.edu.cn/hzhangxyz/pybind11.git /TAT/pybind11 --branch master

RUN cd /TAT/PyTAT && bash ./generator.sh

ADD entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
