from abc import ABC, abstractmethod


class AbstractClassEnvironment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def round(self, pulled_arm):
        pass
