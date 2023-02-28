import plotly.graph_objects as go
import numpy as np

class Visualisation():

    @staticmethod
    def visualise_predicted_trace(prediction:np.ndarray, inputs:np.ndarray, targets:np.ndarray, plot_title=''):
        # TODO visualise predicted trace and targets
        """

        :param prediction: model prediction based on inputs (oy for one trace)
        :param inputs: inputs variables (ox for both)
        :param targets: target variables (oy for one trace)
        :param plot_title: plot title
        """
        figure = go.Figure()

        # в виде точек
        figure.add_trace(
            go.Scatter( 
                x=inputs,
                y=targets,
                mode='markers',
                name='targets'
        ))

        # непрерывная линия
        figure.add_trace(
            go.Scatter(
                x=inputs,
                y=prediction,
                mode='lines',
                name="predictions")
        )

        figure.update_layout(
            title=plot_title,
            xaxis_title="x Axis Title",
            yaxis_title="y Axis Title",
            legend_title="legend",
        )

        figure.show()
        pass

    @staticmethod
    def visualise_error():
        pass