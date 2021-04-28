FROM tensorflow/tensorflow:latest-gpu

WORKDIR /pibic-2020-2021

COPY . /pibic-2020-2021

RUN pip3 install -r requirements.txt
