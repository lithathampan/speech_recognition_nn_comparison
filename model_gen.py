from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM,MaxPool1D)
from tensorflow.keras.initializers import VarianceScaling

def simple_rnn_model(input_dim, output_dim=29,cell=GRU,name='simple_rnn_model'):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = cell(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred,name=name)
    model.output_length = lambda x: x
    print(model.summary())
    #print("Model Detail")
    #print(model.to_yaml())
    return model


def rnn_model(input_dim, units, activation, output_dim=29,name='rnn_model'):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # Add batch normalization 
    bn_rnn = BatchNormalization(axis = -1, name='bn_rnn_1d')(simp_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred, name=name)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29,name='cnn_rnn_model'):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    gru_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # Add batch normalization    
    bn_rnn = BatchNormalization(name='gru_rnn')(gru_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense= TimeDistributed(Dense(output_dim,activation='relu'))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred,name=name)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_pool_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29,name='cnn_rnn_model'):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    #Pooling Layer
    pool_1d = MaxPool1D()(bn_cnn)

    #pool_1d = tf.reshape(pool_1d, [-1, weights['wd1'].get_shape().as_list()[0]])
    #pool_1d = K.reshape(pool_1d, shape=(3,)[0])
    # Add a recurrent layer
    gru_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(pool_1d)
    # Add batch normalization    
    bn_rnn = BatchNormalization(name='gru_rnn')(gru_rnn)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense= TimeDistributed(Dense(output_dim,activation='relu'))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred,name=name)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29, name='deep_rnn_model'):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layers, each with batch normalization
    if recur_layers == 1:
        layer = GRU(units, return_sequences=True, activation='relu')(input_data)
        layer = BatchNormalization(name='bt_rnn_1')(layer)
    else:
        layer = GRU(units, return_sequences=True, activation='relu')(input_data)
        layer = BatchNormalization(name='bt_rnn_1')(layer)

        for i in range(recur_layers - 2):
            layer = GRU(units, return_sequences=True, activation='relu')(layer)
            layer = BatchNormalization(name='bt_rnn_{}'.format(2+i))(layer)

        layer = GRU(units, return_sequences=True, activation='relu')(layer)
        layer = BatchNormalization(name='bt_rnn_last_rnn')(layer)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim,activation='relu'))(layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred, name=name)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29, name='bidirectional_rnn_model'):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, activation='relu'), merge_mode='concat')(input_data)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred, name=name)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29, dropout_rate=0.5, number_of_layers=2, 
    cell=GRU, activation='tanh', name='final_model'):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Specify the layers in your network
    conv_1d = Conv1D(filters, kernel_size, 
                    strides=conv_stride, 
                    padding=conv_border_mode,
                    activation='relu',
                    name='layer_1_conv',
                    dilation_rate=1)(input_data)
    conv_bn = BatchNormalization(name='conv_batch_norm')(conv_1d)


    if number_of_layers == 1:
        layer = cell(units, activation=activation,
            return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)
    else:
        layer = cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)

        for i in range(number_of_layers - 2):
            layer = cell(units, activation=activation,
                        return_sequences=True, implementation=2, name='rnn_{}'.format(i+2), dropout=dropout_rate)(layer)
            layer = BatchNormalization(name='bt_rnn_{}'.format(i+2))(layer)

        layer = cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='final_layer_of_rnn')(layer)
        layer = BatchNormalization(name='bt_rnn_final')(layer)
    
    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred, name=name)
    # Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model