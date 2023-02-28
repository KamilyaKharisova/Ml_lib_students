import plotly.graph_objects as go
import numpy as np


class Visualisation():

    @staticmethod
    def visualise_predicted_trace(prediction: np.ndarray, inputs: np.ndarray, targets: np.ndarray, plot_title=''):
        # TODO visualise predicted trace and targets
        """

        :param prediction: model prediction based on inputs (oy for one trace)
        :param inputs: inputs variables (ox for both)
        :param targets: target variables (oy for one trace)
        :param plot_title: plot title
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=inputs, y=targets, mode='markers', name='gt'))
        fig.add_trace(go.Scatter(x=inputs, y=prediction, name='prediction'))
        fig.update_layout(title=plot_title)
        fig.show()

    @staticmethod
    def visualise_error():
        pass
