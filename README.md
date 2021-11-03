# PIBIC 2020/2021 - Predição de Séries Temporais Baseada em Redes Neurais Artificiais
### Por: João Pagnan

Repositório para guardar os meus arquivos do meu projeto do PIBIC 2020/2021: **Predição de Séries Temporais Baseada em Redes Neurais Artificiais**. O objetivo é o estudo da aplicabilidade de modelos preditores utilizando redes neurais artificiais (especialmente as redes *Multilayer Perceptron* e as recorrentes, como a *Long Short-Term Memory* e as de estado de eco) para a previsão de séries temporais originadas por sistemas com dinâmica caótica.

## Configuração

### Ambiente Virtual para o Python
Você pode utilizar o arquivo **requirements.txt** para criar um ambiente virtual com as bibliotecas necessárias para rodar os códigos, evitando problemas de compatibilidade com as versões locais que você possui em sua máquina. Para criar os ambientes virtuais, recomendo o pacote `python3-venv`, disponível tanto para *Linux* quanto para *Windows*.

Dessa forma, você pode utilizar os seguintes comandos caso você utilizar o `pip`:
```
python3 -m venv <nome_ambiente_virtual> 
source <nome_ambiente_virtual>/bin/activate 
pip3 install -r requirements.txt 
```

Ou também, caso prefira o `conda` os comandos são os seguintes:
```
conda create --prefix ./<nome_ambiente_virtual> --file requirements.txt 
conda activate ./<nome_ambiente_virtual>
```

Vale mencionar que, caso você utilize o *Jupyter Lab* para abrir os arquivos **.ipynb**, é necessário executar os seguintes comandos para habilitar o ambiente virtual dentro dele (considerando que o seu terminal já está dentro do ambiente virtual):
```
ipython kernel install --user --name=<nome_ambiente_virtual>
```
### Docker Container para o sistema
Caso você queira rodar o projeto na sua máquina utilizando o *Tensor Flow* com a sua GPU sem ter muitas dores de cabeça, você pode utilizar o arquivo **Dockerfile** disponibilizado para gerar uma imagem e criar um *container* em que os *drivers* e pacotes necessários para isso sejam instalados e rodem de forma isolada do restante da sua máquina. Isso também evita muito estresse com os problemas de compatibilidade entre as versões do *Tensor Flow*, *drivers* da NVIDIA e versões do seu sistema operacional.

Dessa forma, você rodaria o seguinte comando para criar a imagem:
```
docker build -t <nome_da_imagem> .
```

Em seguida, para rodar um *container* com ela, que será deletado quando o encerrarmos (*flag* `-rm`):
```
docker run -p 8888:8888 --gpus all -it -v $(pwd):/<nome-da-imagem> --rm <nome-da-imagem>
```
Vale mencionar que o **Dockerfile** já contém as instruções para instalar os pacotes necessários para o *Python*, evitando a necessidade de um ambiente virtual.

Essa imagem foi construída em cima da imagem `tensorflow/tensorflow:2.4.1-gpu` e ela é feita para utilizar o *Jupyter Lab*.
Para inicializar o *Jupyter Lab* no *container*:
```
jupyter lab --ip='0.0.0.0' --port=8888 --no-browser --allow-root
```
Em seguida, copie e cole o último *link* fornecido no seu navegador. O *Jupyter Lab* deve abrir.

### Instalação do pacote

Para simplificar a sintaxe, é recomendado instalar o pacote feito por mim para realizar essa pesquisa. É possível fazer isso rodando os seguintes comandos na pasta raiz desse projeto:
```
python3 setup.py install
```

## Estruturação do Repositório
A organização dos diretórios foi baseada na realizada no repositório [eht_imaging](https://github.com/achael/eht-imaging) e na apresentada em [Cookiecutter Data Science](http://drivendata.github.io/cookiecutter-data-science/). 

## Especificações
O projeto está sendo desenvolvido em *Python* na versão **3.8.0**, mais precisamente, rodando num *docker*  cujo sistema operacional *Linux* é a distribuição *Ubuntu 18.04 LTS*. A ferramenta utilizada para o trabalho com as redes neurais foi o *Tensor Flow*, na versão **2.4.1**, utilizando uma *GTX 1070 Ti*, com os *drivers* da *NVIDIA* **460.39**, **Cuda 11.0**, **cuDNN 8.0.4** para auxiliar no treinamento.

## Observação
Como o intuito dos *notebooks* do **Jupyter** era de serem testes exploratórios para aí sim implementar as funções, o código presente neles provavelmente está desatualizado. Se baseie nos códigos presentes nos diretórios do pacote dessa pesquisa caso queira fazer alterações. Além disso, os códigos presentes no pacote estão comentados em português brasileiro (afinal, fiz eles para uso próprio).
