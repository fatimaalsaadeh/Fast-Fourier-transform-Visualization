import math
import wave
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import pyaudio
from matplotlib import gridspec
from scipy.io import wavfile
from test2 import stereoToMono


class Main:
    fr1, fr2, am1, am2 = 0, 0, 0, 0
    sp1, sp2, sp3, sp4 = None, None, None, None
    t, sn = 0, 0
    frames = []

    def fast_fourier_transform(self, is_sbs, samples, root, count):  # FFT algorithm starts here
        n = len(samples)  # length of the samples
        if n == 1:  # base case
            return samples

        # setup to show only one left/right pair at each level of recursion
        pass_on_l = 0
        pass_on_r = 0
        if count == 2:
            pass_on_l = 2
            pass_on_r = 1

        left_side = [num for index, num in enumerate(samples) if index % 2 == 0]  # left side indices
        right_side = [num for index, num in enumerate(samples) if index % 2 == 1]  # right side indices

        transformed_left = self.fast_fourier_transform(is_sbs, left_side, root * root,
                                                       pass_on_l)  # recursive on the left side
        transformed_right = self.fast_fourier_transform(is_sbs, right_side, root * root,
                                                        pass_on_r)  # recursive on the right side

        m_root = 1  # root complex number multiplier for the power of it

        r = [0] * n  # initialize array of zeros
        for j in range(0, int(n / 2)):  # combine
            r[j] = transformed_left[j] + m_root * transformed_right[j]
            r[int(j + n / 2)] = transformed_left[j] - m_root * transformed_right[j]
            m_root = m_root * root
        if count and is_sbs:
            self.sp4.cla()
            ti = self.t[1] - self.t[0]  # Take a discrete interval from the signal time domain
            f = np.linspace(0, 1 / ti, self.sn)  # 1/ti duration signal
            for p1, p2 in zip(f, r):  # Plot the frequency domain and polar
                self.sp2.plot([0, np.angle(p2)], [0, abs(p2)], marker='o')
                self.sp3.bar(p1, abs(p2), width=1.5)  # 1 / N is a normalization factor
                plt.pause(.02)

            self.sp4.scatter([x.real for x in transformed_left], [y.imag for y in transformed_left], alpha=.9)
            self.sp4.scatter([x.real for x in transformed_right], [y.imag for y in transformed_right], alpha=.4)
            plt.pause(.09)
            self.sp2.cla()
            self.sp3.cla()
            self.sp4.cla()
        return r

    def polar_plot(self, cn):  # polar plot of complex number using the angle and
        self.sp2.plot([0, np.angle(cn)], [0, abs(cn)], marker='o')

    def plot_it(self):
        file_name = "voice.wav"
        rate = 44100
        channels = 2

        wf = wave.open(file_name, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        wave_sample_rate, wave_data = wavfile.read('voice.wav')
        wave_data = stereoToMono(wave_data)
        amplitude = wave_data  # (SIGNAL AMPLITUDE)
        fs = wave_sample_rate  # (SAMPLING RATE HZ)
        sn = wave_data.size  # Signal samples
        t = np.linspace(0, sn / fs, sn)  # time of each sample vector

        # Fast Fourier Transform
        fft = self.fast_fourier_transform(False, wave_data, math.e ** (2 * math.pi * 1j / sn), 2)  # DFT
        fft_phase = np.angle(fft)  # (DFT ANGLE rads)
        fft_amplitude = np.abs(fft)  # (DFT AMPLITUDE)
        ti = t[1] - t[0]
        f = np.linspace(0, 1 / ti, sn)  # 1/ti duration signal

        fig2 = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, fig2, bottom=.05)

        sp1 = plt.subplot(gs[0, 0])
        sp1.set_title('Signal')
        sp1.set_ylabel('Amplitude')
        sp1.set_xlabel('Time [s]')
        sp1.plot(t, amplitude)

        sp2 = plt.subplot(gs[1, 0])
        sp2.set_title('Amplitude Spectrum')
        sp2.set_ylabel('Amplitude')
        sp2.set_xlabel('Frequency (Hz)')
        sp2.plot(f, fft_amplitude, color='red')

        sp3 = plt.subplot(gs[0, 1], projection="polar")
        sp3.set_title('Phase Spectrum')
        sp3.set_ylabel('Phase (rads)')
        sp3.set_xlabel('Frequency (Hz)')
        sp3.plot(fft_phase, fft_amplitude, marker='o', color='orange')

        plt.show()

    def start_recording(self):
        frames_per_buffer = 1024
        rate = 44100
        channels = 2
        record_seconds = 3

        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=frames_per_buffer)

        for i in range(0, int(rate / frames_per_buffer * record_seconds)):
            s = stream.read(frames_per_buffer)
            self.frames.append(s)
        stream.stop_stream()
        stream.close()
        p.terminate()

    def audio_processing(self):
        master2 = tk.Tk()
        tk.Button(master2,
                  text='Start Recording',
                  command=self.start_recording).grid(row=4,
                                                     column=1,
                                                     sticky=tk.W,
                                                     pady=4)
        tk.Button(master2,
                  text='Plot It',
                  command=self.plot_it).grid(row=4,
                                             column=2,
                                             sticky=tk.W,
                                             pady=4)

    def final_result(self):
        fs = 500  # (SAMPLING RATE HZ)
        t = np.linspace(0, 250 / fs, fs)  # (SAMPLING PERIOD s)
        ti = t[1] - t[0]
        s = self.am1 * np.sin(self.fr1 * 2 * np.pi * t) + self.am2 * np.sin(self.fr2 * 2 * np.pi * t)  # (SINE WAVE)
        sn = s.size  # Signal samples

        fig2 = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, fig2, bottom=.05)  # Initialize the figure

        plt.figure()

        sp1 = plt.subplot(gs[0, 0])  # row 0, span all columns
        sp1.set_title('Signal')
        sp1.set_ylabel('Amplitude')  # x(t)
        sp1.set_xlabel('Time [s]')
        sp1.grid('on')
        plt.plot(t, s)  # (time, amplitude)

        sp2 = plt.subplot(gs[0, 1], projection='polar')  # row 1, span all columns
        sp2.set_title('Amplitude Spectrum')
        sp2.set_ylabel('Amplitude')  # |X[f]|
        sp2.set_xlabel('Frequency (Hz)')
        sp2.grid('on')

        sp3 = plt.subplot(gs[1,:])  # row 2, span all columns
        sp3.set_title('Phase Spectrum')
        sp3.set_ylabel('Phase (rads)')  # ∠X(k)
        sp3.set_xlabel('Frequency (Hz)')
        sp3.set_xscale('log')
        sp3.grid('on')

        root = math.e ** (2 * math.pi * 1j / sn)  # root of unity
        fft = self.fast_fourier_transform(False, s, root, 2)  # get the signal in frequency domain (DFT)
        f = np.linspace(0, 1 / ti, sn)  # 1/ti duration signal
        for p1, p2 in zip(f, fft):
            sp2.plot([0, np.angle(p2)], [0, abs(p2)], marker='o')
            sp3.bar(p1, abs(p2), width=1.5)  # 1 / N is a normalization factor
        plt.show()

    def step_by_step(self):
        fs = 500  # (SAMPLING RATE HZ)
        self.t = np.linspace(0, 250 / 500, fs)  # (SAMPLING PERIOD s)

        s = self.am1 * np.sin(self.fr1 * 2 * np.pi * self.t) + self.am2 * np.sin(
            self.fr2 * 2 * np.pi * self.t)  # (SINE WAVE)
        self.sn = s.size  # Signal samples

        fig2 = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, fig2)  # Initialize the figure

        plt.figure()
        self.sp1 = plt.subplot(gs[0, 0])  # row 0, span all columns
        self.sp1.set_title('Signal')
        self.sp1.set_ylabel('Amplitude')  # x(t)
        self.sp1.set_xlabel('Time [s]')
        plt.plot(self.t, s)  # (time, amplitude)

        self.sp2 = plt.subplot(gs[0, 1], projection='polar')  # row 1, span all columns
        self.sp2.set_title('Amplitude Spectrum')
        self.sp2.set_ylabel('Amplitude')  # |X[f]|
        self.sp2.set_xlabel('Frequency (Hz)')

        self.sp3 = plt.subplot(gs[1, 0])  # row 2, span all columns
        self.sp3.set_title('Phase Spectrum')
        self.sp3.set_ylabel('Phase (rads)')  # ∠X(k)
        self.sp3.set_yticks(np.arange(-math.pi, math.pi + 1, math.pi / 2))
        self.sp3.set_xlabel('Frequency (Hz)')

        self.sp4 = plt.subplot(gs[1, 1])  # row 2, span all columns
        self.sp4.set_title('FFT step by step')
        self.sp4.set_ylabel('Phase (rads)')  # ∠X(k)
        self.sp4.set_yticks(np.arange(-math.pi, math.pi + 1, math.pi / 2))
        self.sp4.set_xlabel('Frequency (Hz)')

        root = math.e ** (2 * math.pi * 1j / self.sn)  # root of unity
        fft = self.fast_fourier_transform(True, s, root, 2)  # get the signal in frequency domain

    def __init__(self):
        master = tk.Tk()  # create Tkinter object
        master.title("FFT Visualization")
        # input labels
        tk.Label(master, text="Frequency (Hz) Sinusoid 1").grid(row=0)
        tk.Label(master, text="Amplitude Sinusoid 1").grid(row=1)
        tk.Label(master, text="Frequency (Hz) Sinusoid 2").grid(row=2)
        tk.Label(master, text="Amplitude Sinusoid 2").grid(row=3)

        # input rows
        f1 = tk.Entry(master)
        a1 = tk.Entry(master)
        f2 = tk.Entry(master)
        a2 = tk.Entry(master)
        # default values
        f1.insert(10, "40")
        a1.insert(10, "1")
        f2.insert(10, "70")
        a2.insert(10, ".4")

        f1.grid(row=0, column=1)
        a1.grid(row=1, column=1)
        f2.grid(row=2, column=1)
        a2.grid(row=3, column=1)

        self.fr1 = float(f1.get())  # get Frequency 1 in Hz from user input
        self.am1 = float(a1.get())  # amplitude 1
        self.fr2 = float(f2.get())  # get Frequency 2 in Hz from user input
        self.am2 = float(a2.get())  # amplitude 2

        # submit values and begin visualization
        tk.Button(master,
                  text='Step By Step',
                  command=self.step_by_step).grid(row=4,
                                                  column=0,
                                                  sticky=tk.W,
                                                  pady=4)
        tk.Button(master,
                  text='Final Results',
                  command=self.final_result).grid(row=4,
                                                  column=1,
                                                  sticky=tk.W,
                                                  pady=4)
        tk.Button(master,
                  text='Talk to me',
                  command=self.audio_processing).grid(row=4,
                                                      column=2,
                                                      sticky=tk.W,
                                                      pady=4)
        master.mainloop()


if __name__ == '__main__':
    main = Main()
