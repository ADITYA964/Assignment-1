FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y git python3-pip python3.9


ENV MODEL_FILE=model.h5

WORKDIR /app

COPY requirements.txt ./requirements.txt

# Install all need packets


RUN pip3 install -r requirements.txt

COPY . /app

RUN python3 app.py

