import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from data_gen import AudioGenerator
from utils import soundfile_load

class AudioVisualizer:
    def __init__(self,index):
        self.audio_gen_spectro = AudioGenerator(spectrogram=True)
        self.audio_gen_spectro.load_train_data()
        self.vis_text = self.audio_gen_spectro.train_texts[index]
        self.vis_audio_path = self.audio_gen_spectro.train_audio_paths[index]
        self.vis_raw_audio, _ = soundfile_load(self.vis_audio_path)
        self.vis_spectrogram_feature = self.audio_gen_spectro.featurize(self.vis_audio_path)
        self.vis_spectrogram_feature_norm = self.audio_gen_spectro.normalize(self.vis_spectrogram_feature)
        self.audio_gen_mfcc = AudioGenerator(spectrogram=False)
        self.audio_gen_mfcc.load_train_data()
        self.vis_mfcc_feature = self.audio_gen_mfcc.featurize(self.vis_audio_path)
        self.vis_mfcc_feature_norm = self.audio_gen_mfcc.normalize(self.vis_mfcc_feature)

    def plot_raw_audio(self):
        # plot the raw audio signal
        fig = plt.figure(figsize=(12,3))
        ax = fig.add_subplot(111)
        steps = len(self.vis_raw_audio)
        ax.plot(np.linspace(1, steps, steps), self.vis_raw_audio)
        plt.title('Audio Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()

    def plot_mfcc_feature(self,normalized = True):
        # plot the MFCC feature
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(111)
        featuredata = self.vis_mfcc_feature_norm if normalized else self.vis_mfcc_feature
        im = ax.imshow(featuredata, cmap=plt.cm.get_cmap('jet'), aspect='auto')
        title = 'Normalized MFCC' if normalized else 'MFCC'
        plt.title(title)
        plt.ylabel('Time')
        plt.xlabel('MFCC Coefficient')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_xticks(np.arange(0, 13, 2), minor=False)
        plt.show()
    
    def plot_spectrogram_feature(self,normalized = True):
        # plot the normalized spectrogram
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(111)
        featuredata = self.vis_spectrogram_feature_norm if normalized else self.vis_spectrogram_feature
        im = ax.imshow(featuredata, cmap=plt.cm.get_cmap('jet'), aspect='auto')
        title = 'Normalized Spectrogram' if normalized else 'Spectrogram'
        plt.title(title)
        plt.ylabel('Time')
        plt.xlabel('Frequency')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
