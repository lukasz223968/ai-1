import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasAgg, FigureCanvasTkAgg
import tkinter as tk
from itertools import cycle

class Display:
    def __init__(self, button_command):
        self.root = tk.Tk()

        plot_frame = tk.Frame()
        plot_frame.pack()
        self.input_wrapper = tk.Frame()
        self.input_wrapper.pack()
        no_samples = tk.IntVar(value= 1)
        self.label = tk.Label(self.input_wrapper, text= 'No Samples: ')
        self.label.pack(side= tk.LEFT)
        self.no_samples_entry = tk.Entry(self.input_wrapper, textvariable=no_samples)
        self.no_samples_entry.pack(side=tk.LEFT)
        self.button = tk.Button(self.input_wrapper, command= lambda: button_command(no_samples, self), text= 'Train')
        self.button.pack(side=tk.LEFT)

        self.last_boundary: plt.PolyCollection = None
    def set_neuron(self, n):
        self.neuron = n
    def set_gen(self, gen):
        self.gen = gen
    def display_data(self, *data_classes):
        self.fig = Figure(figsize=(3,3), dpi=100)
        self.plot = self.fig.add_subplot(1, 1, 1)
        formats = ['ro', 'bo', 'go']
        for data, format in zip(data_classes, cycle(formats)):    
            self.plot.plot(data[:,0], data[:,1], format)
        full_data = np.concatenate([x for x in data_classes])
        x_min, x_max, self.y_min, y_max = full_data[:,0].min(), full_data[:,0].max(), full_data[:,1].min(), full_data[:,1].max()

        # self.plot.set_xlim(x_min, x_max)
        # self.plot.set_ylim(self.y_min, y_max)

        self.plot.set_xlim(-0.1, 1.1)
        self.plot.set_ylim(-0.1, 1.1)
        
        

        canvas = FigureCanvasTkAgg(self.fig, self.root)
        canvas.get_tk_widget().pack()
    def show_end_of_samples_msgbox():
        tk.messagebox.showwarning(title="End of samples", message="No more samples avaiable")
    def display_decision_boundary(self, weights):
        if self.last_boundary:
            self.last_boundary.remove()
        xs = np.linspace(-2, 2, 10)
        bias = weights[0]
        w1 = weights[1]
        w2 = weights[2]
        a = - (bias / w2) / (bias / w1)
        b = -bias/w2
        y = [a*x + b for x in xs]
        print(y)
        self.last_boundary = self.plot.fill_between(xs, y, -1.5)
        self.fig.canvas.draw()
    def show(self):
        self.root.mainloop()