FROM quay.io/pypa/manylinux2014_x86_64:latest

RUN yum install lapack-devel --assumeyes

ADD entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
