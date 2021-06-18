# tensorflow e keras
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
from tensorflow import keras

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../model-cfg/mlp-basica')

# conjunto de k
import k_set

# modelo
import mlp_model as mlp

# sistemas caoticos
import logisticmap as logistic
import lorenzsystem as lorenz
import henonmap as henon
import mackeyglassequations as mackeyglass

# pega as configuracoes
import mlp_basica_henon as cfg_henon
import mlp_basica_logistic as cfg_logistic
import mlp_basica_lorenz as cfg_lorenz
import mlp_basica_mackeyglass as cfg_mackeyglass

# ---- avaliando para o mapa de henon ---- #

# pegando o conjunto de K's
K_set = k_set.K_henon

# para salvar os resultados
results = np.array([])

# para rodar os K's
for K in K_set:
    # construindo o modelo para esse caso
    model = mlp.ModeloMLP(K, name=cfg_henon.name)
    model.criar_modelo(batch_normalization=cfg_henon.batch_normalization,
                        activation=cfg_henon.activation, init_mode=cfg_henon.init_mode,
                        n_neurons=cfg_henon.n_neurons, n_hidden_layers=cfg_henon.n_hidden_layers)
    model.montar(learning_rate=cfg_henon.learning_rate)

    mse_mean, mse_stddev = model.avaliar(batch_size=cfg_henon.batch_size)

# pegando o conjunto de K's 
K_set = k_set.K_logistic

# pegando o conjunto de K's
K_set = k_set.K_lorenz

# pegando o conjunto de K's
K_set = k_set.K_mackeyglass