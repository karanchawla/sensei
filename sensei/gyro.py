import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from helpers import convert_to_si_units


class Gyroscope(object):
    def __init__(self, params):
        self._params = params

        self._static_bias = np.zeros((1, 3))
        self._previous_dynamic_bias = np.zeros((1, 3))
        self._prev_angle_random_walk_noise = np.zeros((1, 3))
        self._prev_rrw_noise = np.zeros((1, 3))

        self.dt = 1.0 / self._params["frequency"]

        self._initialize_sensor()

    def _initialize_sensor(self):
        self._compute_static_bias()

    def _compute_static_bias(self):
        a = -self._params["static_bias"]
        b = self._params["static_bias"]

        self._static_bias = np.random.uniform(a, b)

    def _compute_white_noise(self):
        white_noise = self._params["noise_density"] * np.random.randn(1, 3)
        return white_noise

    def _compute_dynamic_bias(self):
        dynamic_bias = np.zeros((1, 3), dtype=object)
        # Gauss-Markov process
        for i in range(3):
            beta = self.dt / self._params["correlation_time"][i]
            sigma = self._params["dynamic_bias"][i]
            a1 = np.exp(-beta)
            a2 = sigma * np.sqrt(1 - np.exp(-2 * beta))

            dynamic_bias[0][i] = a1 * \
                self._previous_dynamic_bias[0][i] + a2 * np.random.normal(0, 1)

        self._previous_dynamic_bias = dynamic_bias
        return dynamic_bias

    def _compute_angle_random_walk_noise(self):
        angle_random_walk_noise = np.zeros((1, 3), dtype=object)

        for i in range(3):
            angle_random_walk_noise[0][i] = self._prev_angle_random_walk_noise[0][i] + np.sqrt(
                (1 / self._params["frequency"])) * self._params["angle_random_walk"][i] \
                * np.random.normal(0, 1)
        self._prev_angle_random_walk_noise = angle_random_walk_noise
        return angle_random_walk_noise

    def tick(self, ref):
        simulated_rates = np.empty((1, 3), dtype=object)

        dyn_bias = self._compute_dynamic_bias()
        white_noise = self._compute_white_noise()
        arw_noise = self._compute_angle_random_walk_noise()
        simulated_rates = ref + self._static_bias + dyn_bias + arw_noise + white_noise

        return simulated_rates, arw_noise, dyn_bias, white_noise


def generate_reference_trajectory():
    sample_rate = 100
    start_time = 0
    end_time = 20
    time = np.arange(start_time, end_time, 1 / sample_rate)
    amplitude = 0.1
    theta = 0
    frequency = 0.1
    sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)
    return np.column_stack((sinewave, sinewave, sinewave)), time


if __name__ == '__main__':
    np.random.seed(0)
    pio.templates['shahin'] = pio.to_templated(go.Figure().update_layout(
        margin=dict(t=0, r=0, b=40, l=40))).layout.template
    pio.templates.default = 'shahin'

    ref_traj, time = generate_reference_trajectory()

    # ADIS16488 IMU error profile
    gyro_params = {"frequency": 100.0,                        # Hz
                   "correlation_time": 100.0 * np.ones(3),    # seconds
                   "static_bias": 0.2,                        # deg/s
                   "dynamic_bias": 6.25 / 3600 * np.ones(3),  # deg/s
                   "angle_random_walk": 0.3 * np.ones(3),     # deg/root-hour
                   "noise_density": 0.16                      # deg/sec/root-Hz
                   }

    sanitized_params = convert_to_si_units(gyro_params)

    gyro = Gyroscope(sanitized_params)
    simulated_data = np.empty(np.shape(ref_traj))
    arw_noise = np.empty(np.shape(ref_traj))
    dyn_bias = np.empty(np.shape(ref_traj))
    white_noise = np.empty(np.shape(ref_traj))

    for i in range(np.shape(ref_traj)[0]):
        simulated_data[i][:], arw_noise[i][:], dyn_bias[i][:], white_noise[i][:] = gyro.tick(
            ref_traj[i])

    fig = go.Figure(layout=dict(xaxis=dict(title='Time (sec)'),
                    yaxis=dict(title='Amplitude')))

    fig.add_scatter(x=time, y=np.rad2deg(
        simulated_data[:, 0]), fillcolor='blue')
    # fig.add_scatter(x=time, y=np.rad2deg(ref_traj[:, 0]), fillcolor='red')
    fig.show()
