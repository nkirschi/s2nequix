import jax.numpy as jnp


class EarlyStopping:
    def __init__(
        self,
        patience: int,
        min_relative_improvement: float,
    ):
        self.patience = patience
        self.__best_loss = float("inf")
        self.__loss_at_last_reset = float("inf")
        self.__counter = 0
        self.__min_relative_improvement = min_relative_improvement

    def stop(self, loss: float) -> bool:
        """
        Returns True if the loss is NaN or if the loss is not improving
        for `patience` epochs. Otherwise, returns False.^
        """
        if jnp.isnan(loss):
            print("Loss is NaN, stopping early")
            return True
        if loss < self.__best_loss:
            if loss < self.__loss_at_last_reset * (1 - self.__min_relative_improvement):
                self.__loss_at_last_reset = loss
                self.__counter = 0
            self.__best_loss = loss
            return False
        else:
            self.__counter += 1
            print(f"Early stopping counter: {self.__counter} / {self.patience}")
            return self.__counter >= self.patience
