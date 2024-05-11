from kymatio import TimeFrequencyScattering
import librosa as lr
import openl3
import numpy as np
import torch
import os
import sys
from path import CAE_PATH
sys.path.append(CAE_PATH)
from complex_auto.cqt import standardize
from complex_auto.complex import Complex


class AudioProcessor:

    def __init__(self, sr):
        self.sr = sr

    def __call__(self, audio_array):
        if not isinstance(audio_array, np.ndarray):
            raise ValueError("Input must be a numpy array")
        features = self.compute_features(audio_array)
        return features
    
    def compute_features(self, audio_array):
        raise NotImplementedError("Subclass must implement this method")


class OpenL3(AudioProcessor):

    def __init__(self, sr, hop_size, embedding_size=512):
        super().__init__(sr)
        self.hop_size = hop_size
        self.embedding_size = embedding_size

    def compute_features(self, audio_array):
        emb, ts = openl3.get_audio_embedding(audio_array, self.sr, 
                                             hop_size=self.hop_size,
                                             embedding_size=self.embedding_size,
                                             center=False,
                                             verbose=False)
        return emb.T # time dimension is the last dimension after transposing
    

class MFCC(AudioProcessor):

    def __init__(self, sr, hop_length, n_fft, n_mfcc=20):
        super().__init__(sr)
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc

    def compute_features(self, audio_array):
        mfcc = lr.feature.mfcc(y=audio_array, sr=self.sr, 
                               hop_length=self.hop_length, 
                               n_fft=self.n_fft, 
                               n_mfcc=self.n_mfcc)
        return mfcc


class jTFS(AudioProcessor):

    def __init__(self, sr, J, Q, J_fr, Q_fr, T, F):
        super().__init__(sr)
        self.J = J
        self.Q = Q
        self.J_fr = J_fr
        self.Q_fr = Q_fr
        self.T = T
        self.F = F

    def compute_features(self, audio_array):
        jtfs = TimeFrequencyScattering(shape=(len(audio_array),), 
                                       J=self.J, Q=self.Q, 
                                       J_fr=self.J_fr, Q_fr=self.Q_fr, 
                                       T=self.T, F=self.F, 
                                       format='joint')
        return jtfs(audio_array)
    

class CAE(AudioProcessor):

    def __init__(self, sr, model_save_fn):
        if sr != 22050:
            raise ValueError("CAE only supports 22050 Hz sampling rate")
        super().__init__(sr)
        # cqt params
        self.n_bins = 120
        self.bins_per_oct = 24
        self.fmin = 65.4
        self.hop_length = 1984
        self.length_ngram = 32
        # load model
        self.n_bases = 256
        self._load_model(model_save_fn)
        
    def _load_model(self, model_save_fn):
        self.model = Complex(self.n_bins * self.length_ngram, self.n_bases)
        self.model.load_state_dict(torch.load(model_save_fn, map_location='cpu'), 
                                   strict=False)
        self.model.eval()

    def compute_features(self, audio_array):
        cqt = lr.cqt(audio_array, sr=self.sr, n_bins=self.n_bins, 
                     bins_per_octave=self.bins_per_oct, 
                     fmin=self.fmin, hop_length=self.hop_length)
        mag = lr.magphase(cqt)[0]
        mag = standardize(mag, axis=0)
        repr = mag.transpose()
        ngrams = []
        for i in range(0, len(repr) - self.length_ngram):
            curr_ngram = repr[i:i + self.length_ngram].reshape((-1,))
            curr_ngram = standardize(curr_ngram)
            ngrams.append(curr_ngram)
        x = torch.FloatTensor(np.vstack(ngrams))
        amp, phase = self.model(x)
        return np.array([amp.t().detach().numpy(), 
                         phase.t().detach().numpy()])
        

if __name__ == "__main__":
    
    # test code
    sr = 22050
    audio_array = np.random.randn(22050*20)
    openl3_processor = OpenL3(sr, hop_size=0.1)
    features = openl3_processor(audio_array)
    print(features.shape)
    mfcc_processor = MFCC(sr, hop_length=2048, n_fft=2048)
    features = mfcc_processor(audio_array)
    print(features.shape)
    jtfs_processor = jTFS(sr, J=6, Q=(12,2), J_fr=4, Q_fr=4, T=2**6, F=2**4)
    features = jtfs_processor(audio_array)
    print(features.shape)
    cae_processor = CAE(sr, model_save_fn='../model/model_complex_auto_cqt.save')
    features = cae_processor(audio_array)
    print(features.shape)