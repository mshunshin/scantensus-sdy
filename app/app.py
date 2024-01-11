import gradio as gr

from src.sdy_file import SDYFile
from dataclasses import dataclass, field

from loguru import logger

from tempfile import _TemporaryFileWrapper
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import resample_poly

EXAMPLE_FILE = "sample_data/sample.sdy"
SAMPLE_SDY = SDYFile(EXAMPLE_FILE)

# MARK - App State

@dataclass
class AppState:
    # Inputs

    sdy_file: SDYFile | None

    # the index of the X axis for the flow plot rendering
    flow_plot_x_axis_index: int = 0

    # Outputs

    sdy_exports: list[str] = field(default_factory=list)

# MARK: Functions

def curve_to_csv(data: np.ndarray, time: np.ndarray, filename: str):
    """
    Saves the curve data to a CSV file.

    :param data: curve data
    :param time: time axis
    :param filename: filename to save to
    """
    np.savetxt(filename, np.vstack((time, data)).T, delimiter=',', header='time,flow', comments='')


def render_spectrum_with_flow_tracing(sdy: SDYFile, split_index: int = 0):
    """
    Renders the spectrum with flow tracing in the respecting interval.

    :param sdy: SDYFile object
    :param interval: tuple of start and end index of the interval
    """

    # divide flow into subarrays of 400 elements
    split_flow = np.array_split(sdy.flow, sdy.flow.shape[0] // 400) 
    split_spectrum = np.array_split(sdy.spectrum, sdy.spectrum.shape[1] // 400, axis=1)

    spectrum_img = split_spectrum[split_index]
    spectrum_img = np.sqrt(spectrum_img * 2) * 1.5

    flow = split_flow[split_index]

    time_axis = generate_time_axis(flow, frequency=200)

    plt.imshow(spectrum_img, origin='lower')
    plt.plot(flow)

def export_curves(sdy: SDYFile) -> list[str]:
    """
    Exports the curves to CSV files.

    :param sdy: SDY file
    """    
    curve_to_csv(sdy.flow, generate_time_axis(sdy.flow, frequency=200), 'out/flow.csv')
    curve_to_csv(sdy.pa, generate_time_axis(sdy.pa, frequency=200), 'out/pa.csv')
    curve_to_csv(sdy.pd, generate_time_axis(sdy.pd, frequency=200), 'out/pd.csv')

    return ['out/flow.csv', 'out/pa.csv', 'out/pd.csv']


def generate_time_axis(data: np.ndarray, frequency: float) -> np.ndarray:
    """
    Generates the time axis for a given data array and frequency.

    :param data: data array
    :param frequency: frequency of the data
    :return: time axis
    """

    return np.arange(data.shape[0]) / frequency


def resample(data: np.ndarray, original_frequency: float, target_frequency: float = 50) -> np.ndarray:
    """
    Resamples the data to the target frequency.

    :param data: data to resample
    :param original_frequency: original frequency of the data
    :param target_frequency: target frequency of the data
    :return: resampled data
    """
    # Calculate resampling ratio
    if original_frequency < target_frequency:
        # Upsampling
        up = target_frequency
        down = original_frequency
    else:
        # Downsampling
        up = original_frequency
        down = target_frequency

    # Resample data
    resampled_data = resample_poly(data, up, down)

    # Create time array
    time = np.arange(resampled_data.shape[0]) / target_frequency

    return resampled_data, time


def _load_model():
    pass

# MARK: UI Actions

def update_interval_plot(state: AppState, x_axis_index: int) -> tuple[AppState, plt.figure]:
    logger.info(f'Updating interval plot with x axis index: {x_axis_index}')

    interval_plot = plt.figure()
    render_spectrum_with_flow_tracing(state.sdy_file, x_axis_index)

    return state, interval_plot

def parse_sdy(state: AppState, file: _TemporaryFileWrapper) -> tuple[AppState, plt.figure, plt.figure, list[str]]:
    logger.info(f'Parsing SDY file: {file.name}')

    sdy = SDYFile(file.name)
    state.sdy_file = sdy

    flow_plot = plt.figure()
    render_spectrum_with_flow_tracing(sdy, 0)

    pressure_time_scale = generate_time_axis(sdy.pa, frequency=200)

    pressures_plot = plt.figure()

    plt.plot(pressure_time_scale, sdy.pa, alpha=0.7)
    plt.plot(pressure_time_scale, sdy.pd, alpha=0.7)

    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (?)')

    state.sdy_exports = export_curves(sdy)

    return state, flow_plot, pressures_plot, state.sdy_exports



# MARK: UI Definition

with gr.Blocks() as demo:
    app_state = gr.State(AppState(sdy_file=SAMPLE_SDY))

    with gr.Row():
        with gr.Column():
            gr.Markdown("# Settings")

            sdy_uploader = gr.File(label="SDY File", file_count='single', type='file')

            run_btn = gr.Button("Predict")

        with gr.Column():
            gr.Markdown("# SDY Attributes")

            with gr.Tabs():
                with gr.Tab(label='Flow Plot'):
                    with gr.Row():
                        flow_x_axis_index_input = gr.Slider(label='X Axis Index', min_value=0, max_value=100, step=1, default_value=0)

                    original_flow_plot = gr.Plot(label='Original Flow Plot')

                with gr.Tab(label='Pressures Plot'):
                    pressures_plot = gr.Plot(label='Pressures Plot')

                with gr.Tab(label='Export'):
                    sdy_export_files = gr.File(label='Exported Values', value=[], type='file', file_count='many')

                    gr.Markdown('All values are exported in their original time scales. Pressures are sampled in 400hz while flow is sampled in 200hz.')

        sdy_uploader.upload(parse_sdy, inputs=[app_state, sdy_uploader], outputs=[app_state, original_flow_plot, pressures_plot, sdy_export_files])

        flow_x_axis_index_input.change(update_interval_plot, inputs=[app_state, flow_x_axis_index_input], outputs=[app_state, original_flow_plot])

demo.launch(debug=True, allowed_paths=['sample_data/', 'out/'])