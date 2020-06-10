from learners.SuperSetContext import SuperSetContext
from utility import estimate_daily_n_click
from configuration.SysConfPricing import SysConfPricing
from utility.estimate_daily_n_click import weight
import numpy as np
from context_feature_generation import generate_context_feature
from environment.PricingEnvironment import PricingEnvironment

def function_which_evaluating_the_loweboound_return_active_context():
    # only to makes sense in the loop
    return 2

if __name__ == "__main__":

    conf = SysConfPricing("/home/orso/Documents/POLIMI/DataIntelligenceApplication/DIA-Project/configuration/")
    arms = conf.get_arms_price()
    arms_user_prob = conf.get_function()

    T, n_experiments = conf.get_experiment_context_info()
    n_click = estimate_daily_n_click.n_click_for_days(T) # TODO remember that in the function the budget is fixed.. must be able to change it in a smart way

    prob_user = weight(np.mean(n_click, axis=0))

    active_context = 1 # TODO decide with which one start
    context = SuperSetContext(generate_context_feature(), arms, active_context)
    #TODO cicliamo gli env con gli indici ma in AggregateLearner la logica è implementata con chiave string
    #TODO bisogna quindi usare un mapping (esempio casereccio sotto)
    map_env = ["woman", "man_eu", "man_usa"] #l'ordine lo impene SYsConfPricing che decide l'ordine la prima volta

    number_week = 3 #decide number of week
    collected_reward_week = []

    for e in range(0, n_experiments):
        environment = [PricingEnvironment(n_arms=len(arms), probabilities=p) for p in arms_user_prob]
        context = SuperSetContext(generate_context_feature(), arms, 0)
        click = np.array([sum(el) for el in n_click])   # total number of user each day
        for week in range(0, number_week):
            context.active_context(function_which_evaluating_the_loweboound_return_active_context())
            # each week we decide the context
            for t in range(0, T):
                env_click = [k for k in range(0, len(n_click[0]))]   #[0,1,2] to loop over the env

                while click[t] != 0:  # the user of the day are not terminated
                    i = np.random.choice(a=env_click, p=prob_user)
                    click[t] += -1
                    # qui sotto avviene il mapping... il tipo di user è identificato da un indice ma AggregateLearner
                    # vuole una chiave.
                    # context chiama pull_arm ma sa il context attivo come attributo quindi non serve passarglielo
                    pulled_arm = context.pull_arm(map_env[i])

                    reward = environment[i].round(pulled_arm)

                    context.update(map_env[i], pulled_arm, reward)

            collected_reward_week.append(context.collected_reward()) # ritornerà il reward del context attivo