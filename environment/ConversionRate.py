import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def parab_curve(p, d, title):
    z = np.polyfit(p, d, 2)
    plot = np.poly1d(z)
    show_curves(p, d, plot, title)
    return plot


def show_curves(p, d, plot, title):
    xp = np.linspace(0, 50, 1000)
    vect = np.zeros(1000)
    for i in range(0, xp.size):
        if plot(xp[i]) < 0:
            vect[i] = 0
        else:
            vect[i] = plot(xp[i])

    plt.plot(p, d, '*', xp, vect[:], '-')
    plt.title(title)
    plt.xlabel("Price")
    plt.ylabel("Coversion rate")
    "plt.xlim(0, 50)"
    "plt.ylim(0, 100)"
    plt.plot(p, d, '*', xp, vect[:] * xp, '-')
    plt.show()


def curve_man_eu():
    # European man
    p = np.array([0.0, 25.0, 50.0])
    d = np.array([100.0, 30.0, 0.0])
    return parab_curve(p, d, "European man")


def curve_man_usa():
    # USA man
    p = np.array([0.0, 25.0, 50.0])
    d = np.array([20.0, 40.0, 0.0])
    return parab_curve(p, d, "USA man")


def curve_woman():
    # Woman
    p = np.array([0.0, 25.0, 50.0])
    d = np.array([60.0, 10.0, 0.0])
    return parab_curve(p, d, "Woman")


def interp_man_eu(pl):
    x = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])
    # y = np.array([70.0, 75.0, 60.0, 50.0, 80.0, 30.0, 15.0, 5.0, 0.0])
    y = np.array([0.0, 40.0, 50.0, 70.0, 90.0, 35.0, 20.0, 15.0, 2.0])
    return show_interp(x, y, "European man", "Coversion rate", pl)


def interp_man_usa(pl):
    x = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])
    # y = np.array([50.0, 50.0, 60.0, 30.0, 15.0, 10.0, 5.0, 2.0, 0.0])
    y = np.array([0.0, 70.0, 75.0, 90.0, 85.0, 80.0, 70.0, 35.0, 5.0])
    return show_interp(x, y, "USA man", "Coversion rate", pl)


def interp_woman(pl):
    x = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])
    # y = np.array([30.0, 30.0, 20.0, 20.0, 5.0, 2.0, 11.0, 20.5, 0.0])
    y = np.array([0.0, 90.0, 95.0, 80.0, 20.0, 10.0, 5.0, 2.0, 0.0])
    return show_interp(x, y, "Woman", "Coversion rate", pl)


def interp_marginal_profit(pl):
    x = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])
    y = np.array([-15.0, -9.0, 0.0, 5.0, 10.0, 25.0, 40.0, 35.0, 40.0])
    return show_interp(x, y, "Marginal profit - Price", "Marginal profit", pl)


def show_interp(x, y, title, ylabel, pl):
    x_new = np.linspace(x.min(), x.max(), 200)
    f = interp1d(x, y, kind="quadratic")
    if pl != 0:
        y_smooth = f(x_new)
        plt.plot(x_new, y_smooth)
        plt.scatter(x, y)
        plt.title(title)
        plt.xlabel("Price")
        plt.ylabel(ylabel)
        plt.show()
    return f


def show_total_profit(ratio):
    x_new = np.linspace(interp_man_eu(0).x.min(), interp_man_eu(0).x.max(), 200)
    y_smooth = (ratio[0] * interp_man_eu(0)(x_new) + ratio[1] * interp_man_usa(0)(x_new) +
                ratio[2] * interp_woman(0)(x_new))
    y_smooth_prof = interp_marginal_profit(0)(x_new)
    plt.plot(x_new, y_smooth)
    plt.title("Aggregate conversion function")
    plt.xlabel("Price")
    plt.ylabel("Coversion rate")
    plt.show()
    plt.plot(x_new, y_smooth_prof)
    plt.plot(x_new, y_smooth / 100 * y_smooth_prof)
    plt.title("Aggregate profit")
    plt.xlabel("Price")
    plt.ylabel("Profit")
    plt.legend(["Marginal profit", "Cumulative profit"])
    plt.show()


if __name__ == "__main__":
    interp_man_eu(1)
    interp_man_usa(1)
    interp_woman(1)
    interp_marginal_profit(1)
    show_total_profit([0.1, 0.1, 0.8], "EU", "Conv")
    print((interp_man_eu(0)(19) * 0.1 + interp_man_usa(0)(19) * 0.1 + interp_woman(0)(19) * 0.8) *
          interp_marginal_profit(0)(19))
