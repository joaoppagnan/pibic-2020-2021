# PIBIC 2020/2021 - Predição de Séries Temporais Baseada em Redes Neurais Artificiais
### Por: João Pagnan

Repositório para guardar os meus arquivos do meu projeto do PIBIC 2020/2021: **Predição de Séries Temporais Baseada em Redes Neurais Artificiais**. O objetivo é o estudo da aplicabilidade de modelos preditores utilizando redes neurais artificiais (especialmente as redes *Multilayer Perceptron* e as recorrentes, como a *Long Short-Term Memory* e as de estado de eco) para a previsão de séries temporais originadas por sistemas com dinâmica caótica.

## Configuração
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

## Estruturação do Repositório

A organização dos diretórios foi baseada na realizada no repositório ![eht_imaging](https://github.com/achael/eht-imaging) e na apresentada em ![Cookiecutter Data Science](http://drivendata.github.io/cookiecutter-data-science/). 

## Especificações
O projeto está sendo desenvolvido em *Python*, mais precisamente, a versão **3.8.5** rodando num sistema operacional *Linux*, na distribuição *Ubuntu 20.04 LTS*. A ferramenta utilizada para o trabalho com as redes neurais foi o *Tensor Flow*, na versão **2.4.1**, utilizando uma *GTX 1070 Ti*, com os *drivers* da *NVIDIA* *460.39*, *Cuda 11.0* e *cuDNN 8.0.4* para auxiliar no treinamento.
