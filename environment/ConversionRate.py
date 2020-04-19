import matplotlib.pyplot as plt
import numpy as np


def parab_curve(p, d, title):
    z = np.polyfit(p, d, 2)
    plot = np.poly1d(z)
    xp = np.linspace(0, np.min(plot.roots), 1000)
    plt.plot(p, d, '*', xp, plot(xp), '-')
    plt.title(title)
    plt.xlabel('Price')
    plt.ylabel('Coversion rate')
    plt.xlim(0, 50)
    plt.ylim(0, 100)
    plt.show()


# European man
p = np.array([0.0, 25.0, 50.0])
d = np.array([100.0, 30.0, 0.0])
parab_curve(p, d, "European man")

# USA man
p = np.array([0.0, 25.0, 50.0])
d = np.array([80.0, 20.0, 0.0])
parab_curve(p, d, "USA man")

# Woman
p = np.array([0.0, 25.0, 50.0])
d = np.array([60.0, 10.0, 0.0])
parab_curve(p, d, "Woman")