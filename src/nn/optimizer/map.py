from nn.optimizer.functions import Sgd, Momentum, Nag, Adam, Rmsprop, Nadam


optimizer_map = {
        "sgd": Sgd(),
        "momentum": Momentum(),
        "nag": Nag(),
        "adam": Adam(),
        "rmsprop": Rmsprop(),
        "nadam": Nadam()
    }