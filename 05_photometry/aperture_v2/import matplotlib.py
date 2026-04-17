import matplotlib.pyplot as plt

# Data from 0 to 10
x = range(11)
y = x

plt.plot(x, y)
plt.title("Test Plot: y = x")
plt.xlabel("x-axis")
plt.ylabel("y-axis")

# Essential command to trigger the pop-up window
plt.show()
