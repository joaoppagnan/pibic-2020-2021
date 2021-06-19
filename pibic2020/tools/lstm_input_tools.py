import numpy as np

def formatar_many_to_many(input_vector, L):
    """
    Descrição:
    ----------
    Função para deixar um vetor de entrada no formato Many-to-Many das LSTMs

    Parâmetros:
    -----------
    input_vector: np.ndarray
        Vetor de entrada
    L: int
        Passo de predição

    Retorna:
    --------
    Vetor reformatado para o Many-to-Many

    """

    if not (type(input_vector) is np.ndarray):
        raise TypeError("O vetor deve ser um array do numpy!")

    if not (type(L) is int):
        raise TypeError("O passo de predição deve ser um inteiro!")

    output_array = np.zeros((len(input_vector), L))

    for elemento in range(0, len(input_vector)):
        if (elemento + L < len(input_vector)):
            output_vector = input_vector[elemento:elemento+L]

        # adiciona padding para os últimos instantes de tempo
        else:
            output_vector = input_vector[elemento:len(input_vector)]
            while (output_vector.shape[0] < L):
                output_vector = np.append(output_vector, 0)
        
        output_array[elemento,:] = np.squeeze(output_vector)

    return output_array