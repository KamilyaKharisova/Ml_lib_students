import numpy as np


class LinearRegression:

    def __init__(self, base_functions: list):
        self.weights = np.random.randn(len(base_functions) + 1).reshape(-1, 1)
        self.base_functions = base_functions

    @staticmethod
    def __pseudoinverse_matrix(matrix: np.ndarray) -> np.ndarray:
        """calculate pseudoinverse matrix using SVD. Not this homework """
        pass

    def __plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        inputs = inputs.reshape(-1, 1)
        columns = [np.ones_like(inputs)]
        for func in self.base_functions:
            columns.append(func(inputs))
        return np.hstack(columns)

    def __calculate_weights(self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray) -> None:
        """calculate weights of the model using formula from the lecture. Not this homework"""
        pass

    def calculate_model_prediction(self, plan_matrix) -> np.ndarray:
        return (plan_matrix @ self.weights).flatten()

    def train_model(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """Not this homework"""
        pass

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """return prediction of the model"""
        plan_matrix = self.__plan_matrix(inputs)
        predictions = self.calculate_model_prediction(plan_matrix)

        return predictions
