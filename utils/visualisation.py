import plotly.graph_objects as go
import numpy as np


class Visualisation:

    @staticmethod
    def visualise_predicted_trace(prediction: np.ndarray, inputs: np.ndarray, targets: np.ndarray, plot_title=''):
        prediction = prediction.ravel()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=inputs, y=prediction, mode='lines', name='Prediction'))
        fig.add_trace(go.Scatter(x=inputs, y=targets, mode='markers', name='Target'))
        fig.update_layout(legend_orientation='h',
                          legend=dict(x=.5, xanchor='center', y=1.1),
                          title=plot_title,
                          xaxis_title='inputs',
                          yaxis_title='y')
        fig.show()

    @staticmethod
    def visualise_error():
        pass