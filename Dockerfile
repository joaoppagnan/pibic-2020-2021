FROM tensorflow/tensorflow:2.4.1-gpu

# configura a pasta do docker
WORKDIR /pibic2020-docker
COPY requirements.txt /pibic2020-docker

# conserta o problema da time zone
RUN apt update
ENV TZ=America/Sao_Paulo
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# faz download dos pacotes necessarios
RUN apt install python3.8 -y
RUN python3.8 -m pip install --upgrade pip
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
RUN apt install nodejs -y

# para resolver o sistema de mackey-glass
RUN apt install build-essential libssl-dev libffi-dev python3.8-dev -y

# para poder utilizar o latex no matplotlib
RUN apt install texlive dvipng texlive-latex-extra texlive-fonts-recommended ghostscript cm-super -y

# instala o pacote do python que eu fiz
RUN python3.8 setup.py install 

# libera a porta pro jupyter
EXPOSE 8888
# MD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]
# tirar esse comentario caso queira inicializar o jupyter com o docker
