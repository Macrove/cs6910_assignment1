from nn.loss.functions import CrossEntropyLoss, SquaredErrorLoss

loss_func_map = {
        "cross_entropy": CrossEntropyLoss(),
        "mean_squared_error": SquaredErrorLoss()
    }
