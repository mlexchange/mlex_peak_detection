FROM python:3.9
MAINTAINER TANNY CHAVEZ

COPY docker/requirements.txt requirements.txt 
COPY docker/pyFAI pyFAI

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    python3-pip\
    ffmpeg\
    libsm6\
    libxext6 

RUN pip3 install --upgrade pip &&\
    pip3 install setuptools wheel pip &&\
    pip3 install -r requirements.txt

RUN pip install Cython
#RUN pip install --no-binary=h5py h5py
#RUN pip install silx
#RUN cd /pyFAI &&\
#    python3 pyFAI/setup.py bdist_wheel
#RUN pip install pyFAI --pre --find-links dist --no-index --upgrade


WORKDIR /app/work
ENV HOME /app/work
ENV PYTHONUNBUFFERED=1


COPY block_detection.py block_detection.py
COPY packages packages
CMD ["echo", "running block peak detection"]
