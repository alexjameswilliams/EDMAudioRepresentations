from kymatio import TimeFrequencyScattering
import librosa as lr
import openl3
import numpy as np


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

    def __init__(self, sr):
        super().__init__(sr)

    def compute_features(self, audio_array):
        pass
        

if __name__ == "__main__":
    
    # test code
    sr = 44100
    audio_array = np.random.randn(44100*2)
    openl3_processor = OpenL3(sr, hop_size=0.1)
    features = openl3_processor(audio_array)
    print(features.shape)
    mfcc_processor = MFCC(sr, hop_length=2048, n_fft=2048)
    features = mfcc_processor(audio_array)
    print(features.shape)
    jtfs_processor = jTFS(sr, J=6, Q=(12,2), J_fr=4, Q_fr=4, T=2**6, F=2**4)
    features = jtfs_processor(audio_array)
    print(features.shape)
