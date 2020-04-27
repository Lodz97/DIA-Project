from environment.BudgetEnvironment import BudgetEnvironment


class NonStationaryBudgetEnvironment(BudgetEnvironment):
    """
    A class to represent the Gaussian Process between the number of click w.r.t budgets in a non-stationary situation.
    The bid is fixed
    Attributes
    ----------
    self.__list_clicks_budget : list[] of dictionary {budget: clicks}
        A dictionary for phase
    self.__t_horizon : horizon time
    :raise.__t : int, current round
    """
    def __init__(self, budget, sigma, func, horizon):
        BudgetEnvironment.__init__(self, budget, sigma, func[0])
        self.list_clicks_budget = []
        for x in range(0, len(func)):
            self.list_clicks_budget.append({budget[i]: func[x].apply_func(budget)[i] for i in range(0, len(budget))})
        self.__t_horizon = horizon
        self.__t = 0
        self.__current_phase = 0

    def round(self, pulled_arm):
        """
        Returns a stochastic reward given by the expected number of clicks and the noise of the current phase
        """
        n_phases = len(self.list_clicks_budget)
        phase_size = self.__t_horizon/n_phases
        current_phase = int(self.__t/phase_size)

        if current_phase > self.__current_phase:
            self.clicks_budget = self.list_clicks_budget[current_phase]
            self.__current_phase += 1
        self.__t += 0.5
        return super().round(pulled_arm)

    @property
    def list_clicks_budget(self):
        return self.__list_clicks_budget

    @list_clicks_budget.setter
    def list_clicks_budget(self, list_clicks_budgets):
        self.__list_clicks_budget = list_clicks_budgets






