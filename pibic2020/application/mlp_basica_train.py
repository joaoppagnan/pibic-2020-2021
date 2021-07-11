# mlp_basica_train.py

# bibliotecas gerais
import numpy as np

# bibliotecas de redes neurais e configurações
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
from tensorflow import keras

# modelo
from pibic2020.models import mlp_model

# arquivos de configurações
from pibic2020.parameters.mlp_basica import *

print("Escolha o sistema para treinar a MLP")
print("1: Mapa de Hénon, 2: Mapa Logístico, 3: Sistema de Lorenz, 4: Equações de Mackey-Glass")
sis = input()

# --------------- HENON --------------- #
if (int(sis) == 1):
    from pibic2020.data import henon
    config = mlp_basica_henon.mlp_basica_henon
    model = mlp_model.ModeloMLP(K=4, name=config["name"])

# --------------- LOGISTICO --------------- #
elif (int(sis) == 2):
    from pibic2020.data import logistic

# --------------- LORENZ --------------- #
elif (int(sis) == 3):
    from pibic2020.data import lorenz

# --------------- MACKEYGLASS --------------- #
elif (int(sis) == 4):
    from pibic2020.data import mackeyglass

else:
    print("Comando inválido!")