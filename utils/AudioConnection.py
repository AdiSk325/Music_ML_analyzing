from MA_utils import *

class AudioConn():

    def __init__(self):
        self.HOME_DIR =  '.'
        self.DATA_DIR = os.path.join(self.HOME_DIR, 'data')

    def play_wav(wav_name=None, wf=None, sec=None):

        if (wav_name is None) and (wf is None): 
            print("Podaj ścieżkę wav_name albo wf")
            return None

        p = pyaudio.PyAudio()
        if wf is None: wf = wave.open(wav_name, 'rb')

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        if sec is None: sec = wf.getnframes()/ wf.getframerate()

        CHUNK = 1024
        data = wf.readframes(CHUNK)
        t = time.time()
        sound_time = time.time() - t
        
        continue_stream = True
        while len(data) > 0 and (continue_stream):
            stream.write(data)
            data = wf.readframes(CHUNK)
            sound_time = time.time() - t
            if sound_time > sec:
                continue_stream = False
        
        stream.stop_stream()
        stream.close()

        wf.setpos(0)

        p.terminate()

        return None            

    def play_note(wf=None, note_val=1, note_time=120, note_name=None):

        note_path = os.path.join(self.DATA_DIR,f'notes/{note_name}.wav')
        return play_wav(note_path, wf, sec=(120/(note_time*note_val)))