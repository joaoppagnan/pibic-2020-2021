FROM tensorflow/tensorflow:2.4.1-gpu

WORKDIR /pibic2020-docker
COPY requirements.txt /pibic2020-docker

RUN apt update
RUN apt install python3.8 -y
RUN python3.8 -m pip install --upgrade pip
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 8888
CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]
