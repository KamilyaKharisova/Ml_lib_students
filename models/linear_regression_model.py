import numpy as np


class LinearRegression:
    class LinearRegression():

        def __init__(self, base_functions, reg_coeff):
            self.weights = None
            self.base_functions = base_functions
            self.reg_coeff = reg_coeff

        def __pseudoinverse_matrix(self, matrix):
            U, s, VT = np.linalg.svd(matrix)
            s_inv = np.zeros_like(matrix).T
            s_inv[:len(s), :len(s)] = np.diag(s ** -1)
            return VT.T @ s_inv @ U.T

        def __plan_matrix(self, inputs):
            num_inputs = inputs.shape[0]
            num_functions = len(self.base_functions)
            plan_matrix = np.zeros((num_inputs, num_functions))
            for i, function in enumerate(self.base_functions):
                plan_matrix[:, i] = function(inputs)
            return plan_matrix

        def __calculate_weights(self, plan_matrix, targets):
            self.weights = np.linalg.inv(
                plan_matrix.T @ plan_matrix + self.reg_coeff * np.eye(plan_matrix.shape[1])) @ plan_matrix.T @ targets

        def calculate_model_prediction(self, plan_matrix):
            return plan_matrix @ self.weights

        def train_model(self, inputs, targets):
            plan_matrix = self.__plan_matrix(inputs)
            self.__calculate_weights(plan_matrix, targets)

        def __call__(self, inputs):
            plan_matrix = self.__plan_matrix(inputs)
            predictions = self.calculate_model_prediction(plan_matrix)
            return predictions