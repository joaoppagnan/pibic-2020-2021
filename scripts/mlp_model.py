from tensorflow import keras

class ModeloMLP():
    
    def __init__(self, input_size, name="MLP"):
        """
        Descrição:
        ----------
        Construtor da classe 'ModeloMLP'

        Parâmetros:
        -----------
        input_size: int
            Número de amostras de entrada do preditor
        name: str
            Nome a ser dado para o modelo
        Retorna:
        --------
        Nada
        """
        
        if not (type(input_size) is int):
            raise TypeError("O número de amostras de entrada deve ser um int!")
            
        if not (type(name) is str):
            raise TypeError("O nome do modelo deve ser uma string!")
        
        self._input_size = input_size
        self._name = name
        self.modelo = None
        pass
        
    def construir_mlp(batch_normalization='OFF', activation='selu',
                      init_mode='lecun_normal', n_neurons=30,
                      n_hidden_layers=1):
        """
        Descrição:
        ----------
        Função para gerar a rede neural MLP com os parâmetros especificados
        
        Parâmetros:
        -----------
        batch_normalization: str
            Informa se deve "ON" ou se não deve "OFF" utiizar uma camada de
            batch normalization após a camada de entrada
        activation: str
            Função de ativação a ser utilizada nas camadas intermediárias
        init_mode: str
            Inicialização a ser utilizada para o ajuste dos pesos dos neurônios
        n_neurons: int
            Número de neurônios a ser utilizado na camada intermediária
        n_hidden_layers: int
            Número de camadas intermediárias
        
        Retorna:
        --------
        Nada
        """
        
        if not (type(batch_normalization) is str):
            raise TypeError("A escolha do batch normalization deve ser uma string!")
            
        if not ((batch_normalization == 'ON') | (batch_normalization == 'OFF')):
            raise ValueError("A string do batch normalization deve ser um ON ou OFF")
            
        if not (type(activation) is str):
            raise TypeError("A função de ativação deve ser uma string!")  

        if not (type(init_mode) is str):
            raise TypeError("A inicialização deve ser uma string!")              
        
        if not (type(n_neurons) is int):
            raise TypeError("O número de neurônios deve ser um int!")
            
        if not (type(n_neurons) is int):
            raise TypeError("O número de camadas intermediárias deve ser um int!")     
        
        
        name = self._name
        input_size = self._input_size
        
        # definindo o modelo
        modelo = keras.Sequential(name=name)
        
        # camada de entrada
        modelo.add(keras.layers.Dense(input_size, input_dim=input_size, name="camada_de_entrada", activation = 'linear'))
        
        # camada de batch normalization
        if (batch_normalization == 'ON'):
            modelo.add(keras.layers.BatchNormalization(name="camada_de_batch_normalization"))
        
        # adiciona n_hidden_layers camadas intermediárias
        for i in range(0, n_hidden_layers):
            modelo.add(keras.layers.Dense(n_neurons, input_dim=input_size,
                                          activation=activation, kernel_initializer=init_mode,
                                          name="camada_intermediaria_"+str(i+1)))
        
        # camada de saida
        modelo.add(keras.layers.Dense(1, activation='linear', name="camada_de_saida"))       
    
        self._modelo = modelo
        pass
    
    def compilar(optimizer='Nadam', learning_rate=0.001, loss="mean_squared_error"):
        """
        Descrição:
        ----------
        Função para compilar e montar o modelo com o otimizador selecionado com o learning rate escolhido
        e adotando a função custo
        
        Parâmetros:
        -----------
        optimizer: str
            Otimizador a ser utilizado
        learning_rate: float
            Taxa de aprendizagem
        loss: str
            Função custo
        
        Retorna:
        --------
        Nada
        """
        
        if not(type(optimizer) is str):
            raise TypeError("O otimizador deve ser uma string!")
            
        if not(type(learning_rate) is float):
            raise TypeError("O learning rate deve ser um float!")
            
        if not(type(loss) is str):
            raise TypeError("A função custo deve ser uma string!")
    
        modelo = self._modelo
        
        if (optimizer=='Nadam'):
            model_optimizer = keras.optimizers.Nadam()
        else:
            model_optimizer = optimizer

        model_optimizer.learning_rate.assign(learning_rate)    
            
        modelo.compile(optimizer = model_optimizer,
                       loss = loss)

        modelo.build()
        
        self._modelo = modelo
        pass