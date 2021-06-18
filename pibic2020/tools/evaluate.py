# numpy para uso geral
from typing import Type
import numpy as np
import sklearn

# para fazer as series temporais
from pibic2020.tools import timeseries

# scripts dos modelos
from pibic2020.models import mlp_model
from pibic2020.models import lstm_model
from pibic2020.models import gru_model

def evaluate(modelo, dados, config, k_set, scaler=None, L=3, tam_teste=0.15, tam_val=0.1):
    """
    Descrição:
    ----------
    Função para avaliar um modelo, com base em uma configuração, para um cenário específico
    determinado no conjunto de dados, para certos valores de K e um passo de predição L,
    podendo receber um scaler

    Parâmetros:
    -----------
    modelo: str
        String com o modelo, podendo ser "MLP", "LSTM", ou "GRU"
    dados: np.ndarray
        Array com a série temporal
    config: dict
        Dicionário com as configurações da rede
    L: int
        Passo de predição a ser utilizado
    k_set: np.ndarray
        Conjunto de valores de K para testarr
    tam_teste: float
        Tamanho do conjunto de teste a ser utilizado no treino e teste
    tam_val: float
        Tamanho do conjunto de validação a ser utilizado no treino
    scaler: sklearn.preprocessing._data.MinMaxScaler ou sklearn.preprocessing._data.StandardScaler
        Objeto de scalling do sklearn, sem estar ajustado para os dados

    Retorna:
    --------
    Um array do numpy com os resultados do teste
    """

    if not (type(modelo) is str):
        raise TypeError("O modelo deve ser uma string!")

    if not ((modelo == "MLP") or (modelo == "LSTM") or (modelo == "GRU")):
        raise ValueError("O nome do modelo deve ser um dos mencionados!")
    
    if not (type(dados) is np.ndarray):
        raise TypeError("Os dados devem ser um array do numpy!")
    
    if not ((type(L) is int) and (L > 0)):
        raise TypeError("L deve ser um inteiro maior que zero!")

    if not (type(config) is dict):
        raise TypeError("As configurações devem estar num formato de dicionário!")
    
    if not (type(k_set) is np.ndarray):
        raise TypeError("O conjunto de valores de K deve ser um array do numpy!")

    if ((scaler is not None) and 
        (type(scaler) is not sklearn.preprocessing._data.MinMaxScaler) and
        (type(scaler) is not sklearn.preprocessing._data.StandardScaler)):
        raise TypeError("O scaler deve ser um MinMaxScaler ou StandardScaler!")        

    if not (type(tam_teste) is float):
        raise TypeError("O tamanho do conjunto de teste deve ser um float!")

    if not (type(tam_val) is float):
        raise TypeError("O tamanho do conjunto de validação deve ser um float!")        

    # pega os dados
    x = dados

    # matriz para salvar os resultados
    results = np.zeros((len(k_set), 2))

    # se tiver recebido um scaler, aplica a transformacao nos dados
    if (scaler != None):
        x = scaler.fit_transform(x.reshape(-1,1)).reshape(len(x), )

    # para varrer os K's
    for K in k_set:

        # inicializa o objeto de serie temporal para o K e L dados
        serie_temporal = timeseries.SerieTemporal(x, K+1, L)

        # divide os dados em conjuntos de treino, teste e validação com os parâmetros dados
        X_treino, X_teste, X_val, y_treino, y_teste, y_val = serie_temporal.dividir_treino_teste_validacao(tam_teste, tam_val)

        # inicializa o modelo e configura ele para esse K
        if (modelo=='MLP'):
            model = mlp_model.ModeloMLP(K+1, name=config["name"])
            model.criar_modelo(batch_normalization=config["batch_normalization"],
                                activation=config["activation"],
                                init_mode=config["init_mode"],
                                n_neurons=config["n_neurons"],
                                n_hidden_layers=config["n_hidden_layers"])
            model.montar(learning_rate=config["learning_rate"])

        elif (modelo=='LSTM'):
            model = lstm_model.ModeloLSTM((K+1, 1), name=config["name"])
            model.criar_modelo(n_units=config["n_units"],
                               init_mode=config["init_mode"])
            model.montar(learning_rate=config["learning_rate"])

        elif (modelo=='GRU'):
            model = gru_model.ModeloGRU((K+1, 1), name=config["name"])
            model.criar_modelo(n_units=config["n_units"],
                               init_mode=config["init_mode"])    
            model.montar(learning_rate=config["learning_rate"])        

        # avalia esse modelo e salva os resultados na matriz
        if (modelo=='MLP'):
            mse_mean, mse_stddev = model.avaliar(X_treino, X_teste, X_val,
                                                  y_treino, y_teste, y_val,
                                                  batch_size=config["batch_size"])
        else:
            mse_mean, mse_stddev = model.avaliar(X_treino, X_teste, X_val,
                                                  y_treino, y_teste, y_val,
                                                  batch_size=config["batch_size"],
                                                  scaler=scaler)

        results[K, :] = np.array([mse_mean, mse_stddev])

    return results