
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Create an array of x values from -2 to 2
x = np.linspace(-1, 1, 2000)

# Calculate the y values for each function
y1 = np.tanh(np.tanh(5*(x+1)))
y2 = sigmoid(15 *(-x -0.75))

# Create the plot
plt.figure(figsize=(10, 6))

# Add the functions
plt.plot(x, y1, label='tanh(tanh(5*(x+1)))')
plt.plot(x, y2, label='sigmoid(15*(-x -0.75))')

# Add title and labels
plt.title('')
plt.xlabel('t')
#plt.ylabel('y')

# Add a legend
plt.legend()

# Show the plot
plt.show()
plt.savefig('images/Anatzes.png')