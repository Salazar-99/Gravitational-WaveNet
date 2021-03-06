import tensorflow as tf

#Gravitational-WaveNet
class GWN(tf.keras.Model):
    #Constructor
    def __init__(self, conv_layers=2, filters=[32,32], kernel_size=[4,4], dilation_rate=2, gru_cells=32, **kwargs):
        '''
        Param: conv_layers - Number of stacked convolution layers (int)
        Param: filters - Number of convolutional filters in each convolutional layer (list)
        Param: kernel_size - Width of 1D convolutions in each layer (list)
        Param: dilation_rate - Rate at which successive layers are diluted (int)
        Param: gru_cells - Number of cells in GRU layer (int) 
        '''
        #Inheriting from keras.Model
        super().__init__(**kwargs)
        #First convolutional layer specifies input size
        self.conv_1 = tf.keras.layers.Conv1D(filters=filters[0], 
                                kernel_size=kernel_size[0], 
                                padding='causal', 
                                activation='relu', 
                                input_shape=[None,1])
        #Building the rest of the convolutional layers
        self.conv_layers = []
        for layer in range(1,conv_layers):
            dilation_rate = self.get_dilation_rate(layer, dilation_rate)
            self.conv_layers.append(tf.keras.layers.Conv1D(filters=filters[layer], 
                                                            kernel_size=kernel_size[layer],
                                                            padding='causal',
                                                            activation='relu',
                                                            dilation_rate=dilation_rate))
        #Build GRU layer
        self.gru_layer = tf.keras.layers.GRU(gru_cells, return_sequences=True)
        #Output layer
        self._output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    #Forward pass
    def call(self, inputs):
        conv_1_output = self.conv_1(inputs)
        conv_output = self.prop_through_layers(conv_1_output, self.conv_layers)
        gru_output = self.gru_layer(conv_output)
        _output = self._output(gru_output)
        return _output

    #Helper functions
    def get_dilation_rate(self, depth, dilation_rate):
        return depth**dilation_rate
    
    def prop_through_layers(self, x, layers):
        for layer in layers:
            x = layer(x)
        return x