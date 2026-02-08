FROM ghcr.io/mspass-team/mspass:latest

LABEL maintainer="Ian Wang <yinzhi.wang.cug@gmail.com>"
ENV PFPATH=/test/pf

# Add cxx library
ADD cxx /parallel_pwmig/cxx
ADD data /parallel_pwmig/data
ADD setup.py /parallel_pwmig/setup.py
ADD python /parallel_pwmig/python
# We need python3-dev for Python.h and python3-numpy for numpy/arrayobject.h
USER root
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-numpy \
    cmake \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Now run the install
RUN MSPASS_HOME=/usr/local pip3 install /parallel_pwmig -v
RUN pip3 install vtk

