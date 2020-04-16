from MA_utils import *

class AudioDataAugumentation():
    
    def __init__(self):
        self.name = 'Audio_DA1'
    
    
    def transpose_wav_file(self, path_input, path_output, n_steps):
                           
        x, sr = librosa.load(path_input)
        y = librosa.effects.pitch_shift(x, sr, n_steps=n_steps)
        sf.write(path_output, y, sr, 'PCM_16')
        #print(f'File {path_input} was transposed to file {path_output}')
        
        return None
    
    def chtime_wav_file(self, path_input, path_output, rate):
                           
        x, sr = librosa.load(path_input)
        y = librosa.effects.time_stretch(x, rate)
        sf.write(path_output, y, sr, 'PCM_16')
        #print(f'File {path_input} was chtime to file {path_output}')
        
        return None
        
