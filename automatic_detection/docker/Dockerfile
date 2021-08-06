FROM python:3.9
MAINTAINER TANNY CHAVEZ

COPY docker/requirements.txt requirements.txt 

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


WORKDIR /app/work
ENV HOME /app/work
ENV PYTHONUNBUFFERED=1


COPY block_detection.py block_detection.py
COPY packages packages
CMD ["echo", "running block peak detection"]
