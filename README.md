# NeuralNetworkTester

This tool makes it easy to test the implemented neural network for the classification and regression tasks. Besides using an equation, you can also define your own underlying functions by drawing them.

## Installation

1. Install Python 3 on your machine
2. Download the file main.py and save it at a desired location

## Usage

To use this tool, simply run the file main.py. The following window opens:  
![Main window](https://user-images.githubusercontent.com/10931987/31255236-1fec3022-aa2c-11e7-83fd-e00aa8cf1dfe.png)

In the lower part you can define the parameters of the model you want to run
* **Mode**: Choose whether you want to do a classification or a regression task
* **Data source**: Specify if an equation should be used for the underlying function or whether you want to use your own drawn function
* **Noise (\theta)**: Only accessable if mode is set to "Regression". The training data is gaussian distributed around the true value. Here you specify the standard deviation
* **Number points training**: How many training samples should be generated
* **Number points test**: How many test samples should be generated
* **Time limit (s)**: After your program runs this amount of seconds, it will automatically terminate. Note that this is also what happens on the server used to grade your homework
* **Compiled C program**: Select the **compiled** C program for either classification or regression depending on your earlier choice.
**NOTE**: do not select the C code (a *.c file) here in contrast to the grading system

![Regression result](https://user-images.githubusercontent.com/10931987/31256331-172ed60a-aa32-11e7-9bf6-466533cfc5f9.png)
![Classification result](https://user-images.githubusercontent.com/10931987/31256868-58558f86-aa35-11e7-80b7-f4662bfdf4be.png)

### Equation

If you want to use an equation for the underlying function, choose "Function" for the value of **Data source**.
You specify everything that is related to this function in the upper left area of the window.
* **Function y(x)=**: Give the equation of the function that should be used, e.g. 7+exp(-x/10)*cos(x)**2
  Use Python syntax for the equation. You can use the following functions and constants: *sin*, *asin*, *sinh*, *asinh*, *cos*, *acos*, *cosh*, *acosh*, *tan*, *atan*, *tanh*, *atanh*, *exp*, *log*, *log2*, *log10*, *e*, *pi*
* **x_min**: Start point of the interval within which the function will be evaluated. You can also use constants like *pi*, e.g. *-pi/2*
* **x_max**: End point of the interval within which the function will be evaluated. You can also use constants like *pi*, e.g. *2*pi*
* **y_min**: Only accessable if mode is set to "Classification". Specify the lower bound of the interval within which samples will be generated
* **y_max**: Only accessable if mode is set to "Classification". Specify the upper bound of the interval within which samples will be generated

![Use an equation](https://user-images.githubusercontent.com/10931987/31256189-3b0ac29c-aa31-11e7-9641-193cb31e9d31.png)

### Drawing

You can also draw your own function and use that for the regression or classification task by setting the value of **Data source** to "Drawing".
You specify everything that is related to this function in the upper right area of the window.
* **Main window**: In the large area you draw the function. Simply click and keep the mouse button pressed while you move. Release the button to finish the drawing
* **x_min**: The x-coordinate of the left corners of the drawing area
* **x_max**: The x-coordinate of the right corners of the drawing area
* **y_min**: The y-coordinate of the bottom corners of the drawing area
* **y_max**: The y-coordinate of the upper corners of the drawing area
* **Reset**: Click this button to delete your drawn function and to draw a different one

### Run network

After you have specified all the parameters, you can run the network. Click the button "Run network" at the very bottom of the main window.
Note that the window may seem to be frozen. This does **NOT** mean that something went wrong. Just wait for a response and don't kill the process.
If your program exceeds the time limit or throws and error, you will get notified.
The script also generates a file "input.txt" of the same format that is used on the homework server. This file will be placed in the same directory as the file "main.py"