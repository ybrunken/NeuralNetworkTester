from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import parser
from scipy import interpolate
import numpy as np
import os
import subprocess
import threading
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from math import sin, asin, sinh, asinh, cos, acos, cosh, acosh, tan, atan, tanh, atanh, exp, log, log2, log10, e, pi


DRAW_SIZE_X = 500
DRAW_SIZE_Y = 300


def move_callback(event):
    global x_prev, y_prev

    if x_prev is not None:
        line_id = event.widget.create_line(x_prev, y_prev, event.x, event.y, fill='blue', smooth=TRUE)
        lines.add(line_id)

    x_prev = event.x
    y_prev = event.y
    points.append((event.x, event.y))


def mouse_press_callback(event):
    global x_prev, y_prev, points, lines

    x_prev = None
    y_prev = None
    points = list()
    lines = set()


def mouse_release_callback(event):
    event.widget.unbind("<B1-Motion>")
    event.widget.unbind("<ButtonPress-1>")
    event.widget.unbind("<ButtonRelease-1>")


def reset_drawing():
    global lines, function_canvas

    while len(lines) > 0:
        line_id = lines.pop()
        function_canvas.delete(line_id)

    function_canvas.bind("<B1-Motion>", move_callback)
    function_canvas.bind("<ButtonPress-1>", mouse_press_callback)
    function_canvas.bind("<ButtonRelease-1>", mouse_release_callback)


def set_path():
    global path, path_output

    path = askopenfilename()
    path_output.set(path)


# The following class uses code from stackoverflow.com
# https://stackoverflow.com/a/4825933
# Author: jcollado https://stackoverflow.com/users/183066/jcollado
class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            self.process = subprocess.Popen(self.cmd, shell=True)
            self.process.communicate()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)

        if thread.is_alive():
            self.process.terminate()
            thread.join()
            messagebox.showinfo("Time limit exceeded", "Make sure your program terminates within the given time limit.")


def regression(x_min, x_max, fct):
    # uniformly distribute the training points along the x axis
    train = np.linspace(x_min, x_max, num=int(num_points_training.get()))

    # remove a point with the coordinate zero from training sample to make sure there's no problem with the
    # separation between training and testing
    delete_ind = None
    for i in range(len(train)):
        if train[i] == 0.0:
            delete_ind = i
    if delete_ind:
        train = np.delete(train, delete_ind)

    # shuffle training values
    np.random.shuffle(train)
    # calculate function values of training samples
    if use_drawing.get():
        train_result = fct(train)
    else:
        train_result = list()
        for x in train:
            train_result.append(eval(fct))

    # create some test values
    test = np.random.uniform(x_min, x_max, int(num_points_test.get()))
    # calculate function values of test samples
    if use_drawing.get():
        test_result = fct(test)
    else:
        test_result = list()
        for x in test:
            test_result.append(eval(fct))

    return train, train_result, test, test_result


def classification(x_min, x_max, y_min, y_max, fct):
    # x coordinates for training
    train_x = np.random.uniform(x_min, x_max, int(num_points_training.get()))
    # y coordinates for training
    train_y = np.random.uniform(y_min, y_max, int(num_points_training.get()))
    # function value of the x coordinates
    if use_drawing.get():
        train_value = fct(train_x)
    else:
        train_value = list()
        for x in train_x:
            train_value.append(eval(fct))

    # check if y coordinates of training samples are greater than the function value
    train_comparison = np.greater_equal(train_y, train_value)
    train_comparison = map(lambda elem: 1 if elem else -1, train_comparison)

    # assign points 'above' function either the value '+1' or '-1' randomly
    sign = np.random.choice((-1, 1))
    train_result = map(lambda elem: sign * elem, train_comparison)
    train_result = list(map(lambda elem: '+1' if elem == 1 else '-1', train_result))

    # x coordinates for test
    test_x = np.random.uniform(x_min, x_max, int(num_points_test.get()))
    # y coordinates for training
    test_y = np.random.uniform(y_min, y_max, int(num_points_test.get()))
    # function value of the x coordinates
    if use_drawing.get():
        test_value = fct(test_x)
    else:
        test_value = list()
        for x in test_x:
            test_value.append(eval(fct))

    # check if y coordinates of training samples are greater than the function value
    test_comparison = np.greater_equal(test_y, test_value)
    test_comparison = map(lambda elem: 1 if elem else -1, test_comparison)

    # assign points 'above' function either the value '+1' or '-1' randomly
    test_result = map(lambda elem: sign * elem, test_comparison)
    test_result = list(map(lambda elem: '+1' if elem == 1 else '-1', test_result))

    return train_x, train_y, train_result, test_x, test_y, test_result


def run():
    use_regression_ = use_regression.get()

    if use_drawing.get():
        # values for the bounded drawing area
        x_min = eval(parser.expr(x_min_drawing.get()).compile())
        x_max = eval(parser.expr(x_max_drawing.get()).compile())
        y_min = eval(parser.expr(y_min_drawing.get()).compile())
        y_max = eval(parser.expr(y_max_drawing.get()).compile())

        # coordinates used for interpolation
        interpolation_x = list()
        interpolation_y = list()

        # boundary values for interpolation
        x_min_interpolation = None
        x_max_interpolation = None

        for point in points:
            # transform the points from the drawing area to a coordinate system with the origin in the bottom left
            # and the given dimension
            x_ = x_min + (x_max - x_min) * point[0] / DRAW_SIZE_X
            y_ = y_max + (y_min - y_max) * point[1] / DRAW_SIZE_Y

            if x_min_interpolation is None:
                x_min_interpolation = x_
                x_max_interpolation = x_

            if x_ < x_min_interpolation:
                x_min_interpolation = x_

            if x_ > x_max_interpolation:
                x_max_interpolation = x_

            # make sure there are no duplicate entries with the same x value (moving up/down in straight line)
            if x_ not in interpolation_x:
                interpolation_x.append(x_)
                interpolation_y.append(y_)

        # interpolation function determined with cubic spline
        fct = interpolate.interp1d(interpolation_x, interpolation_y, kind='cubic')

        if use_regression_:
            train, train_result, test, test_result = regression(x_min_interpolation, x_max_interpolation, fct)
        else:
            train_x, train_y, train_result, test_x, test_y, test_result = classification(x_min_interpolation, x_max_interpolation, y_min, y_max, fct)

    else:
        # boundary values for the function evaluation
        x_min = eval(parser.expr(x_min_function.get()).compile())
        x_max = eval(parser.expr(x_max_function.get()).compile())

        # parse function definition
        fct = parser.expr(function.get()).compile()

        # evaluation function for uniformly distributed x values to visualize it
        interpolation_x = np.linspace(x_min, x_max, num=5000)
        interpolation_y = list()
        for x in interpolation_x:
            interpolation_y.append(eval(fct))

        if use_regression_:
            train, train_result, test, test_result = regression(x_min, x_max, fct)
        else:
            # boundary values of the area in which points should be sampled
            y_min = eval(parser.expr(y_min_function.get()).compile())
            y_max = eval(parser.expr(y_max_function.get()).compile())

            train_x, train_y, train_result, test_x, test_y, test_result = classification(x_min, x_max, y_min, y_max, fct)

    if use_regression_:
        if eval(parser.expr(noise.get()).compile()) != 0:
            # add noise
            for i in range(len(train_result)):
                train_result[i] = np.random.normal(train_result[i], eval(parser.expr(noise.get()).compile()))

        with open("input.txt", "w") as inputFile:
            for i in range(len(train)):
                inputFile.write("{},{}\n".format(train[i], train_result[i]))
            inputFile.write("0,0\n")
            for i in range(len(test)):
                if i == len(test) - 1:
                    inputFile.write("{}".format(test[i]))
                else:
                    inputFile.write("{}\n".format(test[i]))
    else:
        with open("input.txt", "w") as inputFile:

            for i in range(len(train_x)):
                inputFile.write("{},{},{}\n".format(train_x[i], train_y[i], train_result[i]))
            inputFile.write("0,0,0\n")
            for i in range(len(test_x)):
                if i == len(test_x) - 1:
                    inputFile.write("{},{}".format(test_x[i], test_y[i]))
                else:
                    inputFile.write("{},{}\n".format(test_x[i], test_y[i]))

    # run C program
    command = Command('"' + os.path.abspath(path) + '" < input.txt > output.txt')
    command.run(timeout=int(timelimit.get()))

    output = list()

    with open("output.txt", "r") as outputFile:
        for value in outputFile:
            if use_regression_:
                output.append(float(value))
            else:
                output.append(int(value))

    if len(output) == 0:
        messagebox.showinfo("Error", "Your program produced no output or crashed!\n"
                                     "You can test your program with this specific input by using the file input.txt "
                                     "in the directory of this Python script.")

    if use_regression_:
        plt.plot(interpolation_x, interpolation_y, 'g', label='underlying function')
        plt.plot(train, train_result, 'ob', markersize=3, label='training')
        plt.plot(test, test_result, 'or', label='test')
        plt.plot(test, output, 'oy', label='prediction')
        plt.legend()
        plt.show()
    else:
        train_positive_x = list()
        train_positive_y = list()
        train_negative_x = list()
        train_negative_y = list()
        for i in range(len(train_result)):
            if train_result[i] == '+1':
                train_positive_x.append(train_x[i])
                train_positive_y.append(train_y[i])
            else:
                train_negative_x.append(train_x[i])
                train_negative_y.append(train_y[i])

        prediction_positive_x = list()
        prediction_positive_y = list()
        prediction_negative_x = list()
        prediction_negative_y = list()
        for i in range(len(output)):
            if output[i] == 1:
                prediction_positive_x.append(test_x[i])
                prediction_positive_y.append(test_y[i])
            else:
                prediction_negative_x.append(test_x[i])
                prediction_negative_y.append(test_y[i])

        plt.plot(interpolation_x, interpolation_y, color='g', label='underlying function')
        plt.plot(train_positive_x, train_positive_y, marker='o', markersize=3, ls='None', color='b', label='positive train')
        plt.plot(train_negative_x, train_negative_y, marker='o', markersize=3, ls='None', color='r', label='negative train')
        plt.plot(prediction_positive_x, prediction_positive_y, marker='o', ls='None', color='deepskyblue', label='positive prediction')
        plt.plot(prediction_negative_x, prediction_negative_y, marker='o', ls='None', color='coral', label='negative prediction')
        plt.legend()
        plt.show()


def change_regression(_, __, ___):
    if use_regression.get():
        y_min_function.configure(state="disabled")
        y_max_function.configure(state="disabled")
        noise.configure(state="normal")
    else:
        y_min_function.configure(state="normal")
        y_max_function.configure(state="normal")
        noise.configure(state="disabled")


def create_function_frame(parent):
    global function, x_min_function, x_max_function, y_min_function, y_max_function

    Label(parent, text="Function y(x)=").grid(row=0, column=0)
    Label(parent, text="x_min").grid(row=1, column=0)
    Label(parent, text="x_max").grid(row=2, column=0)
    Label(parent, text="y_min").grid(row=3, column=0)
    Label(parent, text="y_max").grid(row=4, column=0)

    function = Entry(parent)
    x_min_function = Entry(parent)
    x_max_function = Entry(parent)
    y_min_function = Entry(parent)
    y_max_function = Entry(parent)

    function.grid(row=0, column=1)
    x_min_function.grid(row=1, column=1)
    x_max_function.grid(row=2, column=1)
    y_min_function.grid(row=3, column=1)
    y_max_function.grid(row=4, column=1)


def create_drawing_frame(parent):
    global function_canvas, x_min_drawing, x_max_drawing, y_min_drawing, y_max_drawing

    Label(parent, text="x_min").grid(row=0, column=3)
    Label(parent, text="x_max").grid(row=1, column=3)
    Label(parent, text="y_min").grid(row=2, column=3)
    Label(parent, text="y_max").grid(row=3, column=3)

    x_min_drawing = Entry(parent)
    x_max_drawing = Entry(parent)
    y_min_drawing = Entry(parent)
    y_max_drawing = Entry(parent)

    x_min_drawing.grid(row=0, column=4)
    x_max_drawing.grid(row=1, column=4)
    y_min_drawing.grid(row=2, column=4)
    y_max_drawing.grid(row=3, column=4)

    function_canvas = Canvas(parent, width=DRAW_SIZE_X, height=DRAW_SIZE_Y, bg='white')

    function_canvas.grid(row=0, column=0, columnspan=3, rowspan=6)

    Button(parent, text="Reset", command=reset_drawing).grid(row=5, column=3, columnspan=2, sticky='WE')

    function_canvas.bind("<B1-Motion>", move_callback)
    function_canvas.bind("<ButtonPress-1>", mouse_press_callback)
    function_canvas.bind("<ButtonRelease-1>", mouse_release_callback)


def create_parameter_frame(parent):
    global use_regression, use_drawing, noise, num_points_training, num_points_test, timelimit, path_output

    Label(parent, text="Mode").grid(row=0, column=0)
    Label(parent, text="Data source").grid(row=1, column=0)
    Label(parent, text="Noise (" + u"\u03C3" + ")").grid(row=2, column=0)
    Label(parent, text="Number points training").grid(row=3, column=0)
    Label(parent, text="Number points test").grid(row=4, column=0)
    Label(parent, text="Time limit (s)").grid(row=5, column=0)
    Label(parent, text="Compiled C program").grid(row=6, column=0)
    Label(parent, textvariable=path_output).grid(row=6, column=3)

    use_regression = BooleanVar()
    use_regression.trace("w", change_regression)
    use_drawing = BooleanVar()

    Radiobutton(parent, text="Classification", padx=10, variable=use_regression, value=False).grid(row=0, column=1, sticky='W')
    Radiobutton(parent, text="Regression", padx=10, variable=use_regression, value=True).grid(row=0, column=2, sticky='W')
    Radiobutton(parent, text="Function", padx=10, variable=use_drawing, value=False).grid(row=1, column=1, sticky='W')
    Radiobutton(parent, text="Drawing", padx=10, variable=use_drawing, value=True).grid(row=1, column=2, sticky='W')
    noise = Entry(parent, width=31)
    num_points_training = Entry(parent, width=31)
    num_points_test = Entry(parent, width=31)
    timelimit = Entry(parent, width=31)
    Button(parent, text="Choose file", command=set_path, width=26).grid(row=6, column=1, columnspan=2)

    noise.grid(row=2, column=1, columnspan=2)
    num_points_training.grid(row=3, column=1, columnspan=2)
    num_points_test.grid(row=4, column=1, columnspan=2)
    timelimit.grid(row=5, column=1, columnspan=2)

    noise.insert(0, "0")
    num_points_training.insert(0, "300")
    num_points_test.insert(0, "50")
    timelimit.insert(0, "300")

    noise.configure(state="disabled")

root = Tk()
root.title("NeuralNetworkTester")
root.resizable(False, False)

drawing_frame = LabelFrame(root, text="Drawing", padx=5, pady=5)
function_frame = LabelFrame(root, text="Function", padx=5, pady=5)
parameter_frame = Frame(root)
empty = Label(root, text="").grid(row=1, column=0)

# definition of global variables
function = None
function_canvas = None
x_min_function = None
x_max_function = None
y_min_function = None
y_max_function = None
x_min_drawing = None
x_max_drawing = None
y_min_drawing = None
y_max_drawing = None
use_regression = None
use_drawing = None
noise = None
num_points_training = None
num_points_test = None
timelimit = None
path = None
path_output = StringVar()
x_prev = None
y_prev = None
points = list()
lines = set()

# place the frames
function_frame.grid(row=0, column=0, padx=5)
drawing_frame.grid(row=0, column=1, rowspan=2, padx=5)
parameter_frame.grid(row=2, column=0, columnspan=2, padx=5, sticky=W)

# create frames
create_function_frame(function_frame)
create_drawing_frame(drawing_frame)
create_parameter_frame(parameter_frame)

Button(root, text="Run network", command=run).grid(row=6, column=0, columnspan=2, pady=20, padx=5, sticky='WE')

root.columnconfigure(1, weight=2)
root.rowconfigure(1, weight=2)
drawing_frame.columnconfigure(4, weight=1)
drawing_frame.rowconfigure(4, weight=1)

root.mainloop()
