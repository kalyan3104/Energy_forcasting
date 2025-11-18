import numpy as np

class DummyARIMA:
    def __init__(self, last):
        self.last = float(last)
    def get_forecast(self, steps=1):
        class R:
            def __init__(self, arr):
                self.predicted_mean = np.array(arr)
        arr = [self.last + 0.01*(i+1) for i in range(steps)]
        return R(arr)
