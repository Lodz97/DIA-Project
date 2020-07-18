from learners.SuperSetContext import SuperSetContext
from utility import estimate_daily_n_click
from configuration.SysConfPricing import SysConfPricing
from utility.estimate_daily_n_click import weight
import numpy as np
from context_feature_generation import generate_context_feature
from environment.PricingEnvironment import PricingEnvironment
import plot
import sys
from sklearn import preprocessing

if __name__ == "__main__":
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        print("ERROR: JSON path required")
        sys.exit(1)

    conf = SysConfPricing("/configuration/")
    arms = conf.get_arms_price()
    #arms = np.array(arms)
    #arms = arms / np.linalg.norm(arms)
    #arms = preprocessing.normalize(arms).reshape(-1)
    arms_user_prob = conf.get_function()
    #arms_user_prob = [[0.5, 0.7, 0.9, 0.35, 0.2], [0.75, 0.9, 0.85, 0.8, 0.7], [0.95, 0.8, 0.2, 0.1, 0.05]]
    print(arms_user_prob)

    T, n_experiments, number_week = conf.get_experiment_context_info()
    string_partition = generate_context_feature()

    collected_reward_experiment = []
    n_click = estimate_daily_n_click.n_click_for_days(T)
    click_average = np.array([sum(el) for el in n_click])  # total number of user each day
    prob_user = weight(np.mean(n_click, axis=0))
    print(prob_user)
    #prob_user = [0.30, 0.50, 0.20]

    optimum = []
    opt = []
    for i in range(0, len(prob_user)):
        opt.append(max(np.array(arms_user_prob[i]) * arms))
    opt = dict(zip(["man_eu", "man_usa", "woman"], opt))
    print(opt)
    for e in range(0, n_experiments):
        opt_temp = []
        print("EXP")
        print(e)
        collected_reward = []
        environment = dict(zip(["man_eu", "man_usa", "woman"], [PricingEnvironment(n_arms=len(arms), probabilities=p) for p in arms_user_prob]))
        context = SuperSetContext(string_partition, arms, 0, 0.05)
        for week in range(0, number_week):
            print("WEEK")
            context.print_active_partition()
            click = click_average.copy()
            #click = np.ones(10)*20

            for t in range(0, T):
                click[t] = int(click[t]/10)
                while click[t] != 0:  # the user of the day are not terminated
                    i = np.random.choice(a=string_partition[0], p=prob_user)
                    click[t] += -1

                    pulled_arm = context.pull_arm(i)
                    reward = environment[i].round(pulled_arm)
                    collected_reward.append(reward*arms[pulled_arm])
                    context.update(i, pulled_arm, reward)
                    opt_temp.append(opt[i])
                    
            #if context.active_partition != 4:
            context.select_active_partition()

        collected_reward_experiment.append(collected_reward)
        optimum.append(opt_temp)

    plot.plot_cum_regret(np.array(optimum), np.array(collected_reward_experiment))
    #plot.plot_cum_regret(opt, collected_reward_experiment)
