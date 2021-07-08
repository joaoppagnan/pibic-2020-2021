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
from pibic2020.models import esn_model

def evaluate(modelo, dados, config, k_set, scaler=None, L=3, tam_teste=0.15, tam_val=0.1, verbose=1):
    """
    Descrição:
    ----------
    Função para avaliar um modelo, com base em uma configuração, para um cenário específico
    determinado no conjunto de dados, para certos valores de K e um passo de predição L,
    podendo receber um scaler

    Parâmetros:
    -----------
    modelo: str
        String com o modelo, podendo ser "MLP", "LSTM", "GRU", ou "ESN"
    dados: np.ndarray
        Array com a série temporal
    config: dict
        Dicionário com as configurações da rede
    L: int
        Passo de predição a ser utilizado
    k_set: list
        Conjunto de valores de K para testarr
    tam_teste: float
        Tamanho do conjunto de teste a ser utilizado no treino e teste
    tam_val: float
        Tamanho do conjunto de validação a ser utilizado no treino
    scaler: sklearn.preprocessing._data.MinMaxScaler ou sklearn.preprocessing._data.StandardScaler
        Objeto de scalling do sklearn, sem estar ajustado para os dados
    verbose: int
        Se vai retornar mensagens ao longo do processo (0 ou 1)

    Retorna:
    --------
    Um array do numpy com os resultados do teste
    """

    if not (type(modelo) is str):
        raise TypeError("O modelo deve ser uma string!")

    if not ((modelo == "MLP") or (modelo == "LSTM") or (modelo == "GRU") or (modelo == "ESN")):
        raise ValueError("O nome do modelo deve ser um dos mencionados!")
    
    if not (type(dados) is np.ndarray):
        raise TypeError("Os dados devem ser um array do numpy!")
    
    if not ((type(L) is int) and (L > 0)):
        raise TypeError("L deve ser um inteiro maior que zero!")

    if not (type(config) is dict):
        raise TypeError("As configurações devem estar num formato de dicionário!")
    
    if not (type(k_set) is list):
        raise TypeError("O conjunto de valores de K deve ser uma lista!")

    if ((scaler is not None) and 
        (type(scaler) is not sklearn.preprocessing._data.MinMaxScaler) and
        (type(scaler) is not sklearn.preprocessing._data.StandardScaler)):
        raise TypeError("O scaler deve ser um MinMaxScaler ou StandardScaler!")        

    if not (type(tam_teste) is float):
        raise TypeError("O tamanho do conjunto de teste deve ser um float!")

    if not (type(tam_val) is float):
        raise TypeError("O tamanho do conjunto de validação deve ser um float!")        

    if not ((type(verbose) is int) and
            ((verbose == 0) or
             (verbose == 1))):
        raise ValueError("O valor de verbose deve ser um int igual a 0 ou 1!")

    # pega os dados
    x = dados

    # matriz para salvar os resultados
    results = np.zeros((len(k_set), 3))

    # se tiver recebido um scaler, aplica a transformacao nos dados
    if (scaler != None):
        x = scaler.fit_transform(x.reshape(-1,1)).reshape(len(x), )

    # para varrer os K's
    for K in k_set:

        if (verbose == 1):
            print("Testando para K = " + str(K+1) + "...")

        # inicializa o objeto de serie temporal para o K e L dados
        serie_temporal = timeseries.SerieTemporal(x, K+1, L)

        # divide os dados em conjuntos de treino, teste e validação com os parâmetros dados se for pra MLP, LSTM ou GRU
        if (modelo != "ESN"):
            X_treino, X_teste, X_val, y_treino, y_teste, y_val = serie_temporal.dividir_treino_teste_validacao(tam_teste, tam_val)

        else:
            # ou apenas treino e teste se for a ESN
            X_treino, X_teste, y_treino, y_teste = serie_temporal.dividir_treino_teste(tam_teste)

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

        elif (modelo=='ESN'):
            model = esn_model.ModeloESN(n_neurons=config["n_neurons"], 
                                        spectral_radius=config["spectral_radius"])        

        # avalia esse modelo e salva os resultados na matriz
        if (modelo=='MLP'):
            mse_mean, mse_stddev = model.avaliar(X_treino, X_teste, X_val,
                                                  y_treino, y_teste, y_val,
                                                  batch_size=config["batch_size"])
        elif ((modelo=='LSTM') or (modelo=='GRU')):
            mse_mean, mse_stddev = model.avaliar(X_treino, X_teste, X_val,
                                                  y_treino, y_teste, y_val,
                                                  batch_size=config["batch_size"],
                                                  scaler=scaler)
        elif (modelo=='ESN'):
            mse_mean, mse_stddev = model.avaliar(X_treino, X_teste,
                                                 y_treino, y_teste)

        results[K, :] = np.array([K+1, mse_mean, mse_stddev])

        if (verbose == 1):
            print("Valor Médio do MSE para esse K: " + str(mse_mean))
            print("Desvio Padrão do MSE para esse K: " + str(mse_stddev) + "\n")

    return results