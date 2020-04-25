import numpy as np
from data_gen import AudioGenerator
from tensorflow.keras import backend as K
from utils import int_sequence_to_text
from pprint import pprint
class ModelPredictor:

    def __init__(self,input_to_softmax,model_path):
        self.input_to_softmax = input_to_softmax
        self.input_to_softmax.load_weights(model_path)
        self.audio_path = None
        
    def get_predictions_index(self,index, partition, spectrogram=False):
        """ Print a model's decoded predictions
        Params:
            index (int): The example you would like to visualize
            partition (str): One of 'train' or 'validation'
            input_to_softmax (Model): The acoustic model
            model_path (str): Path to saved acoustic model's weights
        """
        # load the train and test data
        data_gen = AudioGenerator(spectrogram=spectrogram)
        data_gen.load_train_data()
        data_gen.load_validation_data()
        
        # obtain the true transcription and the audio features 
        if partition == 'validation':
            transcr = data_gen.valid_texts[index]
            self.audio_path = data_gen.valid_audio_paths[index]
            data_point = data_gen.normalize(data_gen.featurize(self.audio_path))
        elif partition == 'train':
            transcr = data_gen.train_texts[index]
            self.audio_path = data_gen.train_audio_paths[index]
            data_point = data_gen.normalize(data_gen.featurize(self.audio_path))
        else:
            raise Exception('Invalid partition!  Must be "train" or "validation"')
        

        pprint(data_point)
        # obtain and decode the acoustic model's predictions
        prediction = self.input_to_softmax.predict(np.expand_dims(data_point, axis=0))
        output_length = [self.input_to_softmax.output_length(data_point.shape[0])] 
        pred_ints = (K.eval(K.ctc_decode(
                    prediction, output_length)[0][0])+1).flatten().tolist()
        

        print('True transcription:\n' + '\n' + transcr)
        print('-'*80)
        print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
        print('-'*80)
    
    def get_predictions_recorded(self,spectrogram=False,recordingpath='recordings/demo.wav',):
        """ Print a model's decoded predictions
        Params:
            index (int): The example you would like to visualize
            partition (str): One of 'train' or 'validation'
            input_to_softmax (Model): The acoustic model
            model_path (str): Path to saved acoustic model's weights
        """
        # load the train and test data
        data_gen = AudioGenerator(spectrogram=spectrogram)
        data_gen.load_train_data()
        self.audio_path = recordingpath
        # obtain the true transcription and the audio feature
        data_point = data_gen.normalize(data_gen.featurize(recordingpath))
        pprint(data_point)
        # obtain and decode the acoustic model's predictions
        prediction = self.input_to_softmax.predict(np.expand_dims(data_point, axis=0))
        output_length = [self.input_to_softmax.output_length(data_point.shape[0])] 
        pred_ints = (K.eval(K.ctc_decode(
                    prediction, output_length)[0][0])+1).flatten().tolist()
        print('-'*80)
        print('Predicted transcription:\n' + '\n' + ''.join(int_sequence_to_text(pred_ints)))
        print('-'*80)