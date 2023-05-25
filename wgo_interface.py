import pickle
import random
import abc


class WGOInterface(abc.ABC):
    @abc.abstractmethod
    def wgo(self, clip) -> list[tuple[str, float]]:
        pass


class PickleWGO(WGOInterface):
    def wgo(self, clip) -> list[tuple[str, float]]:
        with open('test.pickle', 'wb') as f:
            pickle.dump(clip, f)
        return list((f'Item {i}', random.random()) for i in range(10))
