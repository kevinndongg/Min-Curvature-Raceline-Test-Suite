from abc import ABC

class Path(ABC):
    @abstractmethod
    def get_position(self, t):
        pass