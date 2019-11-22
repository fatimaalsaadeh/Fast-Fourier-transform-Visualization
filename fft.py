import math
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import sys


# def polar_plot(cn):  # polar plot of complex number using the angle and
#     ax2.plot([0, np.angle(cn)], [0, abs(cn)], marker='o')


def fast_fourier_transform(samples, root, printable, side, layer):  # FFT algorithm starts here
    n = len(samples)  # length of the samples

    if n == 1:  # base case
        return samples

    # setup to show only one left/right pair at each level of recursion
    passonL = 0
    passonR = 0
    if printable == 2:
        passonL = 2
        passonR = 1

    left_side = [num for index, num in enumerate(samples) if index % 2 == 0]  # left side indices
    right_side = [num for index, num in enumerate(samples) if index % 2 == 1]  # right side indices

    transformed_left = fast_fourier_transform(left_side, root * root, passonL, 'Left Side ', layer+1)  # recursive on the left side
    transformed_right = fast_fourier_transform(right_side, root * root, passonR, 'Right Side ', layer+1)  # recursive on the right side

    m_root = 1  # root complex number multiplier for the power of it

    r = [0] * n  # initialize array of zeros
    for j in range(0, int(n / 2)):  # combine
        r[j] = transformed_left[j] + m_root * transformed_right[j]
        r[int(j + n / 2)] = transformed_left[j] - m_root * transformed_right[j]
        m_root = m_root * root

    # plot if designated
    if printable:
        fig.suptitle(side + 'Layer ' + str(layer), size=16)
        ax4.cla()

        ax3.set(xlabel="Frequency [Hz]", ylabel="Amplitude")
        ax3.title.set_text('Current Output Signal in Frequency Domain')

        ax4.title.set_text('Current Left and Right Values')

        if layer == 0:
            ax3.title.set_text('Final Output Signal in Frequency Domain')
        ax4.scatter(transformed_left, np.zeros_like(transformed_left), alpha=.9)
        ax4.scatter(transformed_right, np.zeros_like(transformed_right), alpha=.4)
        plt.tight_layout()
        ti = t[1] - t[0]  # Take a discrete interval from the signal time domain
        f = np.linspace(0, 1 / ti, sn)  # 1/ti duration signal
        for p1, p2 in zip(f, r):  # Plot the frequency domain and polar
            ax2.plot([0, np.angle(p2)], [0, abs(p2)], marker='o')
            ax3.bar(p1, abs(p2), width=1.5)  # 1 / N is a normalization factor
            plt.pause(.0001)

        plt.waitforbuttonpress()
        ax2.cla()
        ax3.cla()
        ax4.cla()
    return r


master = tk.Tk()  # create Tkinter object

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
t = np.linspace(0, 0.5, fs)  # (SAMPLING PERIOD s)

s = am1 * np.sin(fr1 * 2 * np.pi * t) + am2 * np.sin(fr2 * 2 * np.pi * t)  # (SINE WAVE)
sn = s.size  # Signal samples

fig = plt.figure(figsize=(10, 15))  # Initialize the figure
plt.tight_layout()
ax1 = fig.add_subplot(221)  # Initialize the first sine wave plot
ax1.title.set_text('Original Input')
ax1.set(xlabel="Time [s]", ylabel="Amplitude")
ax1.plot(t, s)

ax2 = fig.add_subplot(222, projection='polar')  # Initialize the second polar plot

ax4 = fig.add_subplot(223)  # plot the current left/right values

ax3 = fig.add_subplot(224)  # Initialize the third plot signal in frequency domain
plt.tight_layout()

# set to full screen
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

root = math.e ** (2 * math.pi * 1j / sn)  # root of unity
fft = fast_fourier_transform(s, root, 2, 'Final Output ', 0)  # get the signal in frequency domain


'''
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
'''
