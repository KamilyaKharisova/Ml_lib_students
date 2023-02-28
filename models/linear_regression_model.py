import numpy as np


class LinearRegression:

    def __init__(self, base_functions: list):
        """init weights using np.random.randn (normal distribution with mean=0 and variance=1)"""
        self.weights = np.random.randn(1, len(base_functions))
        self.base_functions = base_functions

    @staticmethod
    def __pseudoinverse_matrix(matrix: np.ndarray) -> np.ndarray:
        """calculate pseudoinverse matrix using SVD. Not this homework """
        pass

    def __plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        """build Plan matrix using list of lambda functions defined in config. Use only one loop (for base_functions)"""
        return np.array([bf(inputs) for bf in self.base_functions]).T

    def __calculate_weights(self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray) -> None:
        """calculate weights of the model using formula from the lecture. Not this homework"""
        pass

    def calculate_model_prediction(self, plan_matrix) -> np.ndarray:
        """calculate prediction of the model (y) using formula from the lecture"""
        return np.dot(self.weights, plan_matrix.T)

    def train_model(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """Not this homework"""
        pass

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """return prediction of the model"""
        plan_matrix = self.__plan_matrix(inputs)
        predictions = self.calculate_model_prediction(plan_matrix)

        return predictions