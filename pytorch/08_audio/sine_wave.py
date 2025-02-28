import numpy
import wave
import struct
from matplotlib import pyplot

sampling_rate = 48000
duration = 1.0
frequency = 440.0
amplitude = 2**15 - 1

t = numpy.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
sine_wave = (amplitude * numpy.sin(2 * numpy.pi * frequency * t)).astype(numpy.int16)

wav_filename = 'sine_wave_440hz.wav'
with wave.open(wav_filename, 'w') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)    # 16-bit (2 bytes)
    wav_file.setframerate(sampling_rate)
    for sample in sine_wave:
        # 以 16-bit 写入
        wav_file.writeframes(struct.pack('<h', sample))

# 仅绘制前 1000 个点
pyplot.plot(t[:1000], sine_wave[:1000])
pyplot.grid()
pyplot.show()
