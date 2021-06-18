# numpy para uso geral
import numpy as np

# para fazer as series temporais
from pibic2020.tools import timeseries

# scripts dos modelos
from pibic2020.models import mlp_model
from pibic2020.models import lstm_model
from pibic2020.models import gru_model

def evaluate(modelo, dados, config, k_set, file_name, scaler=None, L=3, tam_teste=0.15, tam_val=0.1):
    """
    """

    # pega os dados
    x = dados

    # matriz para salvar os resultados
    results = np.array([])

    # se tiver recebido um scaler, aplica a transformacao nos dados
    if (scaler != None):
        x = scaler.fit_transform(x.reshape(-1,1)).reshape(len(x), )

    # para varrer os K's
    for K in k_set:

        # inicializa o objeto de serie temporal para o K e L dados
        serie_temporal = timeseries.SerieTemporal(x, K, L)

        # divide os dados em conjuntos de treino, teste e validação com os parâmetros dados
        X_treino, X_teste, X_val, y_treino, y_teste, y_val = serie_temporal.dividir_treino_teste_validacao(tam_teste, tam_val)

        # inicializa o modelo e configura ele para esse K
        if (modelo=='MLP'):
            model = mlp_model.ModeloMLP(K, name=config["name"])
            model.criar_modelo(batch_normalization=config["batch_normalization"],
                                activation=config["activation"],
                                init_mode=config["init_mode"],
                                n_neurons=config["n_neurons"],
                                n_hidden_layers=config["n_hidden_layers"])
            model.montar(learning_rate=config["learning_rate"])

        elif (modelo=='LSTM'):
            model = lstm_model.ModeloLSTM((K, 1), name=config["name"])
            model.criar_modelo(n_units=config["n_units"],
                               init_mode=config["init_mode"])
            model.montar(learning_rate=config["learning_rate"])

        elif (modelo=='GRU'):
            model = gru_model.ModeloGRU((K, 1), name=config["name"])
            model.criar_modelo(n_units=config["n_units"],
                               init_mode=config["init_mode"])    
            model.montar(learning_rate=config["learning_rate"])        

        # avalia esse modelo e salva os resultados na matriz
        if (modelo=='MLP'):
            mse_mean, mse_stddev = model.avaliar(X_treino, X_teste, X_val,
                                                  y_treino, y_teste, y_val,
                                                  batch_size=config["batch_size"])
            results = np.append(results, [mse_mean, mse_stddev])
        else:
            mse_mean, mse_stddev = model.avaliar(X_treino, X_teste, X_val,
                                                  y_treino, y_teste, y_val,
                                                  batch_size=config["batch_size"],
                                                  scaler=scaler)
            results = np.append(results, [mse_mean, mse_stddev])

    return results