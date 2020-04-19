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
    return plot


def curve_man_eu():
    # European man
    p = np.array([0.0, 25.0, 50.0])
    d = np.array([100.0, 30.0, 0.0])
    return parab_curve(p, d, "European man")

def curve_man_usa():
    # USA man
    p = np.array([0.0, 25.0, 50.0])
    d = np.array([80.0, 20.0, 0.0])
    return parab_curve(p, d, "USA man")

def curve_woman():
    # Woman
    p = np.array([0.0, 25.0, 50.0])
    d = np.array([60.0, 10.0, 0.0])
    return parab_curve(p, d, "Woman")