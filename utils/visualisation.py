import plotly.graph_objects as go
import numpy as np


class Visualisation:

    @staticmethod
    def visualise_predicted_trace(prediction: np.ndarray, inputs: np.ndarray, targets: np.ndarray, plot_title=''):
        """

        :param prediction: model prediction based on inputs (oy for one trace)
        :param inputs: inputs variables (ox for both)
        :param targets: target variables (oy for one trace)
        :param plot_title: plot title
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=inputs,
            y=targets,
            mode='markers',
            name='Фактические наблюдения'
        ))

        fig.add_trace(go.Scatter(
            x=inputs,
            y=prediction,
            name="Предсказания модели",
        ))

        fig.update_layout(
            title=plot_title,
            xaxis_title="X",
            yaxis_title="Y",
            legend_title="Legend",
        )

        fig.show()

    @staticmethod
    def visualise_error():
        pass
