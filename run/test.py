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

    for n in range(0, N_OF_EXPERIMENTS):
        print("EXP")
        print(n)
        pricing_learner = AggregateLearner(key_list=["man_usa", "man_eu", "woman"], arms=pricing_arms, confidence=0,
                                           total_aggregate=True)   # NB confidence is not important for this point
        pricing_env = dict(zip(["man_eu", "man_usa", "woman"],
                           [PricingEnvironment(n_arms=len(pricing_arms), probabilities=p) for p in arms_user_prob]))
        daily_number_click = sum(estimate_daily_n_click.n_click_for_days(1)[0])
        campaign = []
        for idx in range(0, len(budget)):
            campaign.append(BudgetEnvironment(budget[idx], sigma, functions[idx]))

        learners = []
        for k in range(0, len(budget)):
            learners.append(GPTSLearner(len(budget[k]), budget[k], sigma, kernel[k][0], kernel[k][1]))

        comb_learner = CombinatorialLearner(campaign, learners, experiment_params["cum_budget"])

        collected_reward_adv = []

        for day in range(0, T_HORIZON):
            print("DAY")
            print(day)
            # pricing problem
            while daily_number_click != 0:  # the user of the day are not terminated
                i = np.random.choice(a=["man_eu", "man_usa", "woman"], p=user_prob)
                daily_number_click += -1

                pulled_arm = pricing_learner.pull_arm(i)
                reward = pricing_env[i].round(pulled_arm)
                pricing_learner.update(i, pulled_arm, reward)

            value_click = pricing_learner.get_reward_best_arms()
            comb_learner.sc_value_per_click = value_click

            bud_super_arm = comb_learner.knapsacks_solver()
            rewards = comb_learner.get_realization(bud_super_arm)
            comb_learner.update(bud_super_arm, rewards, day)

            daily_number_click = int(sum(rewards))
            user_prob = estimate_daily_n_click.weight(rewards)
            collected_reward_adv.append(sum([rewards[i] * value_click[i] for i in range(0, len(rewards))]))

        combinatorial_reward_experiment.append(collected_reward_adv)