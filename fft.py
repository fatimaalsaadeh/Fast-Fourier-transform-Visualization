import math
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk


def polar_plot(cn):  # polar plot of complex number using the angle and
    ax2.plot([0, np.angle(cn)], [0, abs(cn)], marker='o')


def fast_fourier_transform(samples, root):  # FFT algorithm starts here
    n = len(samples)  # length of the samples

    if n == 1:  # base case
        return samples

    left_side = [num for index, num in enumerate(samples) if index % 2 == 0]  # left side indices
    right_side = [num for index, num in enumerate(samples) if index % 2 == 1]  # right side indices

    #plt.scatter(left_side, np.zeros_like(left_side), alpha=.9)
    #plt.scatter(right_side, np.zeros_like(right_side), alpha=.4)

    #plt.yticks([])
    #plt.show()


    transformed_left = fast_fourier_transform(left_side, root * root)  # recursive on the left side
    transformed_right = fast_fourier_transform(right_side, root * root)  # recursive on the right side

    m_root = 1  # root complex number multiplier for the power of it

    r = [0] * n  # initialize array of zeros
    for j in range(0, int(n / 2)):  # combine
        r[j] = transformed_left[j] + m_root * transformed_right[j]
        r[int(j + n / 2)] = transformed_left[j] - m_root * transformed_right[j]
        m_root = m_root * root

    return r


master = tk.Tk() # create Tkinter object

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

# submit values and begin visualization
tk.Button(master,
          text='Start',
          command=master.destroy).grid(row=4,
                                    column=0,
                                    sticky=tk.W,
                                    pady=4)


fr1 = float(f1.get())  # get Frequency 1 in Hz from user input
am1 = float(a1.get())  # amplitude 1
fr2 = float(f2.get())  # get Frequency 2 in Hz from user input
am2 = float(a2.get())  # amplitude 2
master.mainloop()


fs = 500  # (SAMPLING RATE HZ)
# f = 40  # (Frequency HZ)

t = np.linspace(0, 0.5, fs)  # (SAMPLING PERIOD s)

s = am1 * np.sin(fr1 * 2 * np.pi * t) + am2 * np.sin(fr2 * 2 * np.pi * t)  # (SINE WAVE)
sn = s.size  # Signal samples

root = math.e ** (2 * math.pi * 1j / sn)  # root of unity
fft = fast_fourier_transform(s, root)  # get the signal in frequency domain

fig = plt.figure(figsize=(5, 10))  # Initialize the figure

ax2 = fig.add_subplot(312, projection='polar')  # Initialize the second polar plot

ax1 = fig.add_subplot(311)  # Initialize the first sine wave plot
plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
ax1.plot(t, s)

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
