import numpy as np

class batch_gradient_descent():
    def __init__(self, eta=0.0001):
        self.eta = eta

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])
        
    def get_params(self):
        return {
            "eta": self.eta
        }

    def get_update(self, del_w, del_b):
        return del_w * self.eta, del_b * self.eta


        

        