import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')

from tensorflow import keras

import numpy as np

import sys
sys.path.insert(0, 'model-cfg/gru')
sys.path.insert(0, 'model-cfg/lstm')
sys.path.insert(0, 'model-cfg/mlp-aprimorada')
sys.path.insert(0, 'model-cfg/mlp-basica')
sys.path.insert(0, './')

import k_set
import mlp_model as mlp
import lstm_model as lstm
import gru_model as gru

import logisticmap as logistic
import lorenzsystem as lorenz
import henonmap as henon
import mackeyglassequations as mackeyglass

def evaluatea(arquitetura, cenario):

    # pega o conjunto de K para esse cenario
    if (cenario == 'henon'):
        K_set = k_set.K_henon

    elif (cenario == 'logistic'):
        K_set = k_set.K_logistic

    elif (cenario == 'lorenz'):
        K_set = k_set.K_lorenz

    elif (cenario == 'mackeyglass'):
        K_set = k_set.K_mackeyglass

    # vetor para salvar o resultado para todos os K's
    results = np.array([])

    # para varrer os K's
    for K in K_set:

        # inicializa o objeto para esse K
        if ((arquitetura == 'mlp-basica') or
            (arquitetura == 'mlp-aprimorada')):
            model = mlp.ModeloMLP(K, name=cfg.name)
            model.criar_modelo(batch_normalization=cfg.batch_normalization,
                        activation=cfg.activation, init_mode=cfg.init_mode,
                        n_neurons=cfg.n_neurons, n_hidden_layers=cfg.n_hidden_layers)
            model.montar(learning_rate=cfg.learning_rate)

        elif (arquitetura == 'lstm'):
            model = lstm.ModeloLSTM((K, 1), name=cfg.name)
            model.criar_modelo(n_units = cfg.n_units, init_mode=cfg.init_mode)
            model.montar(learning_rate=cfg.learning_rate)       

        elif (arquitetura == 'gru'):
            model = gru.ModeloGRU((K, 1), name=cfg.name)
            model.criar_modelo(n_units = cfg.n_units, init_mode=cfg.init_mode)
            model.montar(learning_rate=cfg.learning_rate)       
    return