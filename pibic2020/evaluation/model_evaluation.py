# bibliotecas gerais
import numpy as np
import pandas as pd

# bibliotecas de redes neurais e configurações
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
from tensorflow import keras

# arquivos de configurações
from pibic2020.parameters.gru import *
from pibic2020.parameters.lstm import *
from pibic2020.parameters.mlp_aprimorada import *
from pibic2020.parameters.mlp_basica import *

# sistemas caoticos
from pibic2020.data import henon
from pibic2020.data import logistic
from pibic2020.data import lorenz
from pibic2020.data import mackeyglass

modelo = None
while ((modelo != '1') and
       (modelo != '3') and
       (modelo != '2') and
       (modelo != '3')):

    print("Selecione o modelo que você quer avaliar:")
    modelo = input("1: MLP Básica, 2: MLP Aprimorada, 3: LSTM, 4: GRU")
    print("Selecione uma opção válida!")

modelo = int(modelo)

sistema = None
while ((sistema != '1') and
       (sistema != '3') and
       (sistema != '2') and
       (sistema != '3')):

    print("Selecione o cenário que você quer avaliar:")
    modelo = input("1: Mapa de Henon, 2: Mapa Logístico, 3: Sistema de Lorenz, 4: Equações de Mackey-Glass")
    print("Selecione uma opção válida!")
