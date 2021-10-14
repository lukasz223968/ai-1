import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasAgg, FigureCanvasTkAgg
import tkinter as tk

from numpy.core.fromnumeric import shape

def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def generate_single_mode(no_samples):
    rng = np.random.default_rng()
    mean = rng.random() * 20 - 10
    variance = rng.random()

    data = rng.normal(mean, variance, (no_samples, 2))
    return data
def display_single_class(root, no_samples, no_modes, format):
    data = np.array([generate_single_mode(no_samples) for i in range(no_modes)])
    print(data.shape)
    #data = generate_single_mode(no_samples)
    plot(data, root, format)

def display_data(root, no_samples_0, no_modes_0, no_samples_1, no_modes_1):
    
    fig = Figure(figsize=(3,3), dpi=100)
    plot = fig.add_subplot(1, 1, 1)

    display_single_class(plot, no_samples_0, no_modes_0, 'ro')
    display_single_class(plot, no_samples_1, no_modes_1, 'b.')
    
    canvas = FigureCanvasTkAgg(fig, root)
    canvas.get_tk_widget().grid(row=0, column=0)
    
def plot(data, plot, format):
    colors = ['bo', 'ro', 'go']
    for i in range(data.shape[0]):
        plot.plot(data[i, :,0], data[i, :,1], format)
    
def gen_window(parent, label):
    frame = tk.Frame(parent)
    frame.pack(side = tk.LEFT)

    no_samples, no_modes = tk.IntVar(), tk.IntVar()

    label = tk.Label(frame, text=label)
    label.pack()

    samples_input_wrapper = tk.Frame(frame)
    samples_input_wrapper.pack()

    samples_label = tk.Label(samples_input_wrapper, text="Number of samples: ")
    samples_label.pack(side = tk.LEFT)

    samples_input = tk.Entry(samples_input_wrapper, textvariable=no_samples)
    samples_input.pack(side=tk.LEFT)

    mode_input_wrapper = tk.Frame(frame)
    mode_input_wrapper.pack()

    mode_label = tk.Label(mode_input_wrapper, text="Number of modes: ")
    mode_label.pack(side = tk.LEFT)

    mode_input = tk.Entry(mode_input_wrapper, textvariable=no_modes)
    mode_input.pack(side=tk.LEFT)

    return no_samples, no_modes



def main():
    root = tk.Tk()
    input_wrapper = tk.Frame()
    input_wrapper.pack()
    samples_0, modes_0 = gen_window(input_wrapper, "First Class Settings")
    samples_1, modes_1 = gen_window(input_wrapper, "Second Class Settings") 

    plot_frame = tk.Frame()
    plot_frame.pack()

    generate_button = tk.Button(text="Generate plot", command = lambda: display_data(plot_frame, samples_0.get(), modes_0.get(), samples_1.get(), modes_1.get()))
    generate_button.pack(side=tk.BOTTOM)
    root.mainloop()
    pass




if __name__ == '__main__':
    main()
