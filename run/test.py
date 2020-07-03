from environment.ConversionRate import interpolate_curve
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0])
    y_woman = [90.0, 95.0, 80.0, 20.0, 10.0, 5.0, 2.0, 0.0, 0.0]
    y_man_usa = [70.0, 75.0, 90.0, 85.0, 80.0, 70.0, 35.0, 5.0, 0.0]
    y_man_eu = [40.0, 50.0, 70.0, 90.0, 35.0, 20.0, 15.0, 2.0, 0.0]
    prices = [10, 15, 20.0, 25.0, 30.0, 35.0]
    woman = interpolate_curve(x, y_woman)(prices) * 0.01 * prices
    man_usa = interpolate_curve(x, y_man_usa)(prices) * 0.01 * prices
    man_eu = interpolate_curve(x, y_man_eu)(prices) * 0.01 * prices
    plt.plot(prices, woman, "r", label=u"woman")
    plt.plot(prices, man_eu, "g", label=u"man_eu")
    plt.plot(prices, man_usa, "b", label=u"man_usa")
    plt.legend(loc='lower right')
    plt.show()

    #"woman": [90.0, 95.0, 80.0, 20.0, 10.0, 5.0, 2.0, 0.0, 0.0],
    #"man_eu": [40.0, 50.0, 70.0, 90.0, 35.0, 20.0, 15.0, 2.0, 0.0],
    #"man_usa": [70.0, 75.0, 90.0, 85.0, 80.0, 70.0, 35.0, 5.0, 0.0]

    #"woman": [60.0, 70.0, 90.0, 20.0, 10.0, 5.0, 2.0, 0.0, 0.0],
    #"man_eu": [40.0, 50.0, 70.0, 98.0, 35.0, 20.0, 15.0, 2.0, 0.0],
    #"man_usa": [50.0, 60.0, 65.0, 75.0, 80.0, 90.0, 35.0, 5.0, 0.0]"