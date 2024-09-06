FROM python:3.12
RUN apt update
RUN apt install --yes libopenblas-dev libopenmpi-dev libhdf5-openmpi-dev
RUN pip install build torch openfermion
COPY . TNSP
RUN python -m build TNSP/PyTAT -o dist -v
RUN python -m build TNSP/lazy_graph -o dist -v
RUN python -m build TNSP/PyScalapack -o dist -v
RUN python -m build TNSP/tetragono -o dist -v
RUN python -m build TNSP/tetraku -o dist -v
RUN python -m build TNSP/tnsp_bridge -o dist -v
RUN pip install dist/*.whl
