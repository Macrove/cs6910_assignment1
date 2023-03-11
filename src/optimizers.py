import numpy as np

class Sgd():
    def __init__(self, eta=0.00001):
        self.eta = eta

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])
        
        self.reset()
        
    def get_params(self):
        return {
            "eta": self.eta
        }

    def reset(self):
        self.del_w = np.zeros(self.w_shape)
        self.del_b = np.zeros(self.b_shape)

    def get_update(self, w, b):
        w_new = w - self.eta * self.del_w
        b_new = b - self.eta * self.del_b
        return w_new, b_new


class Momentum():
    def __init__(self, eta=0.0001, gamma=0.001):
        self.eta = eta
        self.gamma = gamma

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])
        self.reset()
        
    def get_params(self):
        return {
            "eta": self.eta,
            "gamma": self.gamma
        }

    def reset(self):
        self.del_w = np.zeros(self.w_shape)
        self.del_b = np.zeros(self.b_shape)
        self.w_prev_update = np.zeros(self.w_shape)
        self.b_prev_update = np.zeros(self.b_shape)
        
    def get_update(self, w, b):
        w_new_update = self.gamma * self.w_prev_update + self.eta * self.del_w
        b_new_update = self.gamma * self.b_prev_update + self.eta * self.del_b
        w_new = w - w_new_update
        b_new = b_new_update
        self.w_prev_update = w_new_update
        self.b_prev_update = b_new_update
        return w_new, b_new


class Nag():
    def __init__(self, eta=0.0001, gamma=0.001):
        self.eta = eta
        self.gamma = gamma

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])
        self.reset()
        
    def get_params(self):
        return {
            "eta": self.eta,
            "gamma": self.gamma
        }

    def reset(self):
        self.del_w = np.zeros(self.w_shape)
        self.del_b = np.zeros(self.b_shape)
        self.w_prev_update = np.zeros(self.w_shape)
        self.b_prev_update = np.zeros(self.b_shape)
        

    def get_update(self, w, b):
        
        w_new_update = self.gamma * self.w_prev_update + self.eta * self.del_w
        b_new_update = self.gamma * self.b_prev_update + self.eta * self.del_b

        w_new = w - w_new_update
        b_new = b - b_new_update

        w_look_ahead = w_new - self.gamma * w_new_update
        b_look_ahead = b_new - self.gamma * b_new_update

        self.w_prev_update = w_new_update
        self.b_prev_update = b_new_update

        return w_look_ahead, b_look_ahead


class Rmsprop():
    def __init__(self, eta=0.0001, beta=0.9, epsilon=0.001):
        self.eta = eta
        self.beta = beta
        self.epsilon = epsilon

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])
        self.reset()
        
    def get_params(self):
        return {
            "eta": self.eta,
            "beta": self.beta,
            "epsilon": self.epsilon
        }

    def reset(self):
        self.del_w = np.zeros(self.w_shape)
        self.del_b = np.zeros(self.b_shape)
        self.w_v_prev = np.zeros(self.w_shape)
        self.b_v_prev = np.zeros(self.b_shape)
        

    def get_update(self, w, b):
        
        w_v_new = self.beta * self.w_v_prev + (1 - self.beta) * np.power(self.del_w, 2)
        b_v_new = self.beta * self.b_v_prev + (1 - self.beta) * np.power(self.del_b, 2)

        w_new = w - (self.eta / np.power(w_v_new + self.epsilon, 0.5)) * self.del_w
        b_new = b - (self.eta / np.power(b_v_new + self.epsilon, 0.5)) * self.del_b

        return w_new, b_new


class Adam():
    def __init__(self, eta=0.0001, beta1=0.5, beta2=0.5, epsilon=0.001):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])
        self.reset()
        
    def get_params(self):
        return {
            "eta": self.eta,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon
        }

    def reset(self):
        self.del_w = np.zeros(self.w_shape)
        self.del_b = np.zeros(self.b_shape)

        self.w_m_prev = np.zeros(self.w_shape)
        self.b_m_prev = np.zeros(self.b_shape)

        self.w_v_prev = np.zeros(self.w_shape)
        self.b_v_prev = np.zeros(self.b_shape)

        self.t = 0
        

    def get_update(self, w, b):

        self.t += 1
        
        w_m_new = self.beta1 * self.w_m_prev + (1 - self.beta1) * self.del_w
        b_m_new = self.beta1 * self.b_m_prev + (1 - self.beta1) * self.del_b

        w_v_new = self.beta2 * self.w_v_prev + (1 - self.beta2) * np.power(self.del_w, 2)
        b_v_new = self.beta2 * self.b_v_prev + (1 - self.beta2) * np.power(self.del_b, 2)

        w_m_new_hat = w_m_new / (1 - np.power(self.beta1, self.t))
        b_m_new_hat = b_m_new / (1 - np.power(self.beta1, self.t))

        w_v_new_hat = w_v_new / (1 - np.power(self.beta2, self.t))
        b_v_new_hat = b_v_new / (1 - np.power(self.beta2, self.t))

        w_new = w - (self.eta / np.power(w_v_new_hat + self.epsilon, 0.5)) * w_m_new_hat
        b_new = b - (self.eta / np.power(b_v_new_hat + self.epsilon, 0.5)) * b_m_new_hat

        self.w_v_prev = w_v_new_hat
        self.b_v_prev = b_v_new_hat

        return w_new, b_new


class Nadam():
    def __init__(self, eta=0.0001, beta1=0.5, beta2=0.5, epsilon=0.001):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def set_params(self, params):
        for key in params:
            setattr(self, key, params[key])
        self.reset()
        
    def get_params(self):
        return {
            "eta": self.eta,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon
        }

    def reset(self):
        self.del_w = np.zeros(self.w_shape)
        self.del_b = np.zeros(self.b_shape)

        self.w_m_prev = np.zeros(self.w_shape)
        self.b_m_prev = np.zeros(self.b_shape)

        self.w_v_prev = np.zeros(self.w_shape)
        self.b_v_prev = np.zeros(self.b_shape)

        self.t = 0
        

    def get_update(self, w, b):

        self.t += 1
        
        w_m_new = self.beta1 * self.w_m_prev + (1 - self.beta1) * self.del_w
        b_m_new = self.beta1 * self.b_m_prev + (1 - self.beta1) * self.del_b

        w_v_new = self.beta2 * self.w_v_prev + (1 - self.beta2) * np.power(self.del_w, 2)
        b_v_new = self.beta2 * self.b_v_prev + (1 - self.beta2) * np.power(self.del_b, 2)

        w_m_new_hat = w_m_new / (1 - np.power(self.beta1, self.t))
        b_m_new_hat = b_m_new / (1 - np.power(self.beta1, self.t))

        w_v_new_hat = w_v_new / (1 - np.power(self.beta2, self.t))
        b_v_new_hat = b_v_new / (1 - np.power(self.beta2, self.t))

        w_new = w - (self.eta / np.power(w_v_new_hat + self.epsilon, 0.5)) * (w_m_new_hat + (1 - self.beta1) * self.del_w / (1 - np.power(self.beta1, self.t)))
        b_new = b - (self.eta / np.power(b_v_new_hat + self.epsilon, 0.5)) * b_m_new_hat

        self.w_v_prev = w_v_new_hat
        self.b_v_prev = b_v_new_hat

        return w_new, b_new
