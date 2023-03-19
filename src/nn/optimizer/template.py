import numpy as np

class Optimizer():
    def __init__(self):
        pass

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])
        
        self.reset()
        
    def get_params(self):
        pass

    def reset(self):
        pass

    def get_update(self, w, b):
        return w, b

    def get_partial_update(self, w, b):
        return w, b