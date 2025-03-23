import wave
import matplotlib.pyplot as plt
import numpy
import librosa

with wave.open('res/audio/piano_c4.wav', 'rb') as wav_file:
    num_channels = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()
    num_frames = wav_file.getnframes()
    
    audio_data = wav_file.readframes(num_frames)
    audio_array = numpy.frombuffer(audio_data, dtype=numpy.int16)
    

plt.plot(audio_array)
plt.title('Waveform of the audio')
plt.xlabel('Sample index')
plt.ylabel('Amptitude')
plt.grid()
plt.show()

n_fft = 2048
audio_array_float = audio_array.astype(numpy.float32) / numpy.iinfo(numpy.int16).max
ft = numpy.abs(librosa.stft(audio_array_float[:n_fft], hop_length=n_fft + 1))
plt.plot(ft)
plt.title('Spectrum')
plt.xlabel('Frequency bin')
plt.ylabel('Amplitude')
plt.show()

spec = numpy.abs(librosa.stft(audio_array_float, hop_length=512))
spec = librosa.amplitude_to_db(spec, ref=numpy.max)
librosa.display.specshow(spec, sr=frame_rate, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()