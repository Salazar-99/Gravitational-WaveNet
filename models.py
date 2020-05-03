import tensorflow as tf

#CNN stacked on top of GRU
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
        #Set input shape of batch_size=1, unspecified sequence length, 1D data
        self._input = tf.keras.layers.InputLayer(input_shape=([1,None,1]))
        #Build convolutional layers
        self.conv_layers = []
        for layer in range(conv_layers):
            dilation_rate = self.get_dilation_rate(layer, dilation_rate)
            self.conv_layers.append(tf.keras.layers.Conv1D(filters=filters[layer], 
                                                            kernel_size=kernel_size[layer],
                                                            padding='causal',
                                                            activation='relu',
                                                            dilation_rate=dilation_rate))
        #Build GRU layer
        self.gru_layer = tf.keras.layers.GRU(gru_cells, return_sequences=True)
        #Output layer
        self._output = tf.keras.layers.Dense(1, activation='sigmoid')
    
    #Forward pass
    def call(self, inputs):
        inputs = self._input(inputs)
        conv_output = self.prop_through_layers(inputs, self.conv_layers)
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