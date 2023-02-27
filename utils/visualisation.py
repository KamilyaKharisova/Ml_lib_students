import plotly.graph_objects as go
import numpy as np


class Visualisation:

    @staticmethod
    def visualise_predicted_trace(prediction: np.ndarray, inputs: np.ndarray, targets: np.ndarray, plot_title=''):
        """visualise predicted trace and targets
        :param prediction: model prediction based on inputs (oy for one trace)
        :param inputs: inputs variables (ox for both)
        :param targets: target variables (oy for one trace)
        :param plot_title: plot title
        """

        prediction = prediction.ravel()  # return a contiguous flattened array
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=inputs, y=prediction, mode='lines', name='prediction'))
        fig.add_trace(go.Scatter(x=inputs, y=targets, mode='markers', name='target values'))
        fig.update_layout(legend_orientation='h',
                          legend=dict(x=.5, xanchor='center'),
                          title=plot_title,
                          xaxis_title='inputs',
                          yaxis_title='y')
        fig.show()

    @staticmethod
    def visualise_error():
        pass
