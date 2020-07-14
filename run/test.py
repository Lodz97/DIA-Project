
import numpy as np
from learners.GPTSLearner import GPTSLearner
import matplotlib.pyplot as plt
import BiddingEnv
import GPTS
def fun(x):
    return 100 * (1.0 - np.exp(-4*x + 3*x**3))


if __name__ == '__main__':
    """x = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0])
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
    plt.show()"""

    #"woman": [90.0, 95.0, 80.0, 20.0, 10.0, 5.0, 2.0, 0.0, 0.0],
    #"man_eu": [40.0, 50.0, 70.0, 90.0, 35.0, 20.0, 15.0, 2.0, 0.0],
    #"man_usa": [70.0, 75.0, 90.0, 85.0, 80.0, 70.0, 35.0, 5.0, 0.0]

    #"woman": [60.0, 70.0, 90.0, 20.0, 10.0, 5.0, 2.0, 0.0, 0.0],
    #"man_eu": [40.0, 50.0, 70.0, 98.0, 35.0, 20.0, 15.0, 2.0, 0.0],
    #"man_usa": [50.0, 60.0, 65.0, 75.0, 80.0, 90.0, 35.0, 5.0, 0.0]"

    n_arms = 20
    min_bid = 0.0
    max_bid = 1.0
    bids = np.linspace(min_bid, max_bid, n_arms)
    sigma = 10

    T = 60
    n_experiments = 100
    gpts_reward_per_experiment = []

    for e in range (0, n_experiments):
        env = BiddingEnv.BiddingEnv(bids, sigma)
        gpts_learner = GPTS.GPTS(n_arms, bids)

        for t in range(0,T):
            pulled_arm = gpts_learner.pull_arm()
            print(pulled_arm)
            reward = env.round(pulled_arm)
            gpts_learner.update(pulled_arm,reward)

        gpts_reward_per_experiment.append(gpts_learner._collected_rewards)

    opt = np.max(env.means)
    plt.figure(0)
    plt.plot(np.cumsum(np.mean(opt - gpts_reward_per_experiment, axis=0)),'r')
    plt.show()