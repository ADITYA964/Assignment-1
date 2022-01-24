ARG IMAGE_VARIANT=slim-buster
ARG OPENJDK_VERSION=8
ARG PYTHON_VERSION=3.9.8


FROM python:${PYTHON_VERSION}-${IMAGE_VARIANT} AS py3
FROM openjdk:${OPENJDK_VERSION}-${IMAGE_VARIANT}

COPY --from=py3 / /

ARG PYSPARK_VERSION=3.2.0
RUN pip --no-cache-dir install pyspark==${PYSPARK_VERSION}

WORKDIR /app

ENV VER=1
ENV RAND=123
ENV N_SAMP=20000

COPY requirements2.txt ./requirements2.txt

# Install all need packets

RUN pip3 install -r requirements2.txt --user

COPY . /app

RUN python3 spark_preprocess.py

