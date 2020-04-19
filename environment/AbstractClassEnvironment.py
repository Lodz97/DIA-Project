from abc import ABC, abstractmethod


class AbstractClassEnvironment(ABC):
    """
    The class which represents the environment of our experiments
    """
    def __init__(self):
        pass

    @abstractmethod
    def round(self, pulled_arm):
        pass
