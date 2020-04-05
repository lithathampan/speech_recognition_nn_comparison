"""
Defines a functions for training a NN.
"""

from data_gen import AudioGenerator
import _pickle as pickle
from tensorflow import keras
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Lambda)
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.callbacks import ModelCheckpoint  ,TensorBoard
from tensorflow.keras import losses 
from pprint import pprint
import datetime
import os

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    #print(args)
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def add_ctc_loss(input_to_softmax):
    the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)
    # CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [input_to_softmax.output, the_labels, output_lengths, label_lengths])
    model = Model(
        inputs=[input_to_softmax.input, the_labels, input_lengths, label_lengths], 
        outputs=loss_out)
    return model
def train_model(input_to_softmax, 
                pickle_path,
                save_model_path,
                train_json='train_corpus.json',
                valid_json='valid_corpus.json',
                minibatch_size=20,
                spectrogram=True,
                mfcc_dim=13,
                #optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),
                #optimizer=SGD()
                optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.9999, amsgrad=False),
                epochs=20,
                verbose=1,
                sort_by_duration=False,
                max_duration=10.0):
    
    # create a class instance for obtaining batches of data
    audio_gen = AudioGenerator(minibatch_size=minibatch_size, 
        spectrogram=spectrogram, mfcc_dim=mfcc_dim, max_duration=max_duration,
        sort_by_duration=sort_by_duration)
    # add the training data to the generator
    audio_gen.load_train_data(train_json)
    audio_gen.load_validation_data(valid_json)
    # calculate steps_per_epoch
    num_train_examples=len(audio_gen.train_audio_paths)
    steps_per_epoch = num_train_examples//minibatch_size
    # calculate validation_steps
    num_valid_samples = len(audio_gen.valid_audio_paths) 
    validation_steps = num_valid_samples//minibatch_size
    
    # add CTC loss to the NN specified in input_to_softmax
    model = add_ctc_loss(input_to_softmax)

    # CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer,metrics=[metrics.mae,metrics.binary_crossentropy])
    #model.compile(loss=losses.mean_squared_error, optimizer=optimizer,metrics=[metrics.mae,metrics.binary_crossentropy])

    # make results/ directory, if necessary
    if not os.path.exists('results'):
        os.makedirs('results')

    # add checkpointer
    checkpointer = ModelCheckpoint(filepath='results/'+save_model_path, verbose=0)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # train the model
    hist = model.fit(x=audio_gen.next_train(), steps_per_epoch=steps_per_epoch,
        epochs=epochs, validation_data=audio_gen.next_valid(), validation_steps=validation_steps,
        callbacks=[checkpointer,tensorboard_callback], verbose=verbose)
    #pprint(audio_gen.next_train())
    audio_gen.load_test_data('test_corpus.json')
    score = model.evaluate(x=audio_gen.next_test(), verbose=1,steps =steps_per_epoch)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    '''    .fit(x=, y=,
          batch_size=None, epochs=,
          verbose=1, validation_data=None,
          steps_per_epoch=None, validation_steps=None,
          validation_batch_size=None, validation_freq=1)

    .compile(optimizer='adam', loss=None, metrics=['accuracy'], loss_weights=None,
                  sample_weight_mode=None, weighted_metrics=None)'''
    # save model loss
    with open('results/'+pickle_path, 'wb') as f:
        pickle.dump(hist.history, f)