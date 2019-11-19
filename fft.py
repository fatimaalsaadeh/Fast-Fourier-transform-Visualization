import math
import matplotlib.pyplot as plt
import numpy as np
import tkinter


fig = plt.figure(figsize=(5, 10))  # Initialize the figure

ax2 = fig.add_subplot(312, projection='polar')  # Initialize the second polar plot


def polar_plot(cn):  # polar plot of complex number using the angle and
    ax2.plot([0, np.angle(cn)], [0, abs(cn)], marker='o')


def fast_fourier_transform(samples, root):  # FFT algorithm starts here
    n = len(samples)  # length of the samples

    if n == 1:  # base case
        return samples

    left_side = [num for index, num in enumerate(samples) if index % 2 == 0]  # left side indices
    right_side = [num for index, num in enumerate(samples) if index % 2 == 1]  # right side indices

    transformed_left = fast_fourier_transform(left_side, root * root)  # recursive on the left side
    transformed_right = fast_fourier_transform(right_side, root * root)  # recursive on the right side

    m_root = 1  # root complex number multiplier for the power of it

    r = [0] * n  # initialize array of zeros
    for j in range(0, int(n / 2)):  # combine
        r[j] = transformed_left[j] + m_root * transformed_right[j]
        r[int(j + n / 2)] = transformed_left[j] - m_root * transformed_right[j]
        m_root = m_root * root

    return r


fs = 500  # (SAMPLING RATE HZ)
f = 40  # (Frequency HZ)
t = np.linspace(0, 0.5, fs)  # (SAMPLING PERIOD s)
s = np.sin(f * 2 * np.pi * t)  # (SINE WAVE)
sn = s.size  # Signal samples
ax1 = fig.add_subplot(311)  # Initialize the first sine wave plot
plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
ax1.plot(t, s)

root = math.e ** (2 * math.pi * 1j / sn)  # root of unity
fft = fast_fourier_transform(s, root)  # get the signal in frequency domain

ti = t[1] - t[0]  # Take a discrete interval from the signal time domain
f = np.linspace(0, 1 / ti, sn)  # 1/ti duration signal

ax3 = fig.add_subplot(313)  # Initialize the third plot signal in frequency domain
plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")

for f1, f2 in zip(f, fft):  # Plot the frequency domain and polar
    [polar_plot(f2)]
    ax3.bar(f1, abs(f2), width=1.5)  # 1 / N is a normalization factor
    plt.pause(.02)

plt.show()
