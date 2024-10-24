from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import random
import eventlet

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import windows, decimate, find_peaks, welch, butter, sosfilt
from rtlsdr import RtlSdr
from collections import deque
from matplotlib.animation import FuncAnimation
import time
import threading
import multiprocessing as mp
from multiprocessing import Manager
import queue
from scipy.signal import butter, filtfilt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

manager = Manager()
shared_data = manager.dict()


# Create Flask app
app = Flask(__name__)
socketio = SocketIO(app)

# Use RtlSdrTcpClient to connect to the remote rtl_tcp server
sdr = RtlSdr()

# Configure SDR parameters
sdr.sample_rate = 2.56e6  # 2.048 MSPS
sdr.center_freq = 1420.405e6   # 1.42 GHz center frequency
sdr.gain = 48          # Automatic gain control
sdr.bias_tee = True

# Number of samples per read and sliding window size
num_samples = 1 * 1024

num_intermediate_avg = 1000  # Number of FFT frames for intermediate averaging

shared_data['dark_frame_state'] = "Initial"
shared_data['dark_frame_status'] = "Dark Frame Empty"
shared_data['integration_minutes'] = 3
shared_data['process_settings_state'] = "Nothing"

shared_data['update_rtlsdr_settings_state'] = "Nothing"
shared_data['sdr_sample_rate'] = sdr.sample_rate

gain = 1
decimation_factor = 2  # Decimation factor to reduce the sampling rate
nperseg = num_samples * 2 // decimation_factor  # Set nperseg to any value you want

# Define FFT size for zero-padding to increase resolution
fft_size = num_samples * 1  # Zero-pad to increase FFT resolution
x_data = np.linspace(-sdr.sample_rate / (2 * decimation_factor), sdr.sample_rate / (2 * decimation_factor), nperseg)



# Multiprocessing queue to hold incoming samples for processing
sample_queue = mp.Queue(maxsize=1024)
matrix_sample_queue = mp.Queue(maxsize=64)
plot_queue = mp.Queue(maxsize=32)  # Queue to send data to the plotter



# Callback function to collect the samples
def callback(samples, context):
    # Add samples to the multiprocessing queue
    try:
        sample_queue.put(samples, block=False)
        #print("put samples")
    except mp.queues.Full:
        # Drop the sample if the queue is full
        print("dropping samples")
        pass


# Process num_intermediate_avg samples into a matrix
def process_sample_matrix(sample_queue):
    while(True):
        # Create an empty list to collect samples for matrix processing
        samples_list = []

        # Process samples in batches
        for _ in range(num_intermediate_avg):
            # Try to collect samples from the queue
            #try:
            samples = sample_queue.get()
            #except queue.Empty:
            #    continue

            # Append samples to the list for later matrix processing
            samples_list.append(samples)

        # Convert the list of samples to a matrix
        # Each row in the matrix represents one set of samples
        samples_matrix = samples_list
        
        # Send the result to the plotting queue
        try:
            matrix_sample_queue.put(samples_matrix, block=False)
        except mp.queues.Full:
            print("dropping sample matrix")
            pass    


# Function to update the plot
def process_samples(matrix_sample_queue):
    num_cumulative_avg = int((shared_data['integration_minutes'] * 60) / ((1.0 / shared_data['sdr_sample_rate']) * num_samples * num_intermediate_avg))
    cumulate_buffer = deque(maxlen=num_cumulative_avg)
    dark_frame = None   
    dark_frame_counter = 0

    while True:
        match shared_data['process_settings_state']:
            case "update_cumulate_buffer": # Reset and resize the cumulute buffer if a integration_minutes, num_samples or sample_rate changes
                num_cumulative_avg = int((shared_data['integration_minutes'] * 60) / ((1.0 / shared_data['sdr_sample_rate']) * num_samples * num_intermediate_avg))  # Number of intermediate averages for cumulative averaging
                cumulate_buffer = deque(maxlen=num_cumulative_avg)
                shared_data['process_settings_state'] = "Nothing"
                
        print(shared_data['sdr_sample_rate'])
        # Perform intermediate averaging in a for loop
        #intermediate_buffer = None
        intermediate_avg_count = 0
        
        print(num_cumulative_avg)
        
        start_time = time.time()
        samples_matrix = matrix_sample_queue.get()
        
         # Apply decimation to the entire matrix if needed
        if decimation_factor > 1:
            decimated_matrix = decimate(samples_matrix, decimation_factor, axis=0, ftype='fir')  # Decimate along the sample axis (columns)
        else:
            decimated_matrix = samples_matrix

        # Apply Welch's method to the entire matrix
        # Welch will compute the PSD for each row (each set of samples)
        f, Pxx_matrix = welch(
            decimated_matrix, 
            fs=shared_data['sdr_sample_rate'] / decimation_factor, 
            nperseg=nperseg, 
            axis=1,  # Apply Welch along the sample axis (columns)
            return_onesided=False, 
            scaling='spectrum'
        )

        # Shift PSD results for centered frequency representation
        Pxx_matrix = np.fft.fftshift(Pxx_matrix, axes=1)

        # Initialize intermediate buffer for accumulation, if it's not already initialized
        #if intermediate_buffer is None:
        #    intermediate_buffer = np.zeros(Pxx_matrix.shape[0])  # Buffer matches the shape of one FFT output (across columns)

        # Accumulate FFT magnitudes in the intermediate buffer (averaging across columns)
        intermediate_avg  = np.mean(Pxx_matrix, axis=0)  # Average across all columns (frequencies)
        

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.6f} seconds")

        #######################################################################
        # Fix the DC spike problem with welch
        dc_index = nperseg // 2 #np.argmin(np.abs(f))  # The index where frequency is closest to 0
        
        interp_dist = 3;
        sample_dist = 5;
        # Interpolate around the DC component
        val_mid = (intermediate_avg[dc_index - sample_dist] + intermediate_avg[dc_index + sample_dist]) / 2

        noise_std = np.std([intermediate_avg[dc_index - sample_dist], intermediate_avg[dc_index + sample_dist]])
        # Perform linear interpolation to estimate the DC component
        intermediate_avg[dc_index-interp_dist:dc_index+interp_dist] = val_mid + np.random.normal(0, noise_std, size=interp_dist*2)   # Interpolate at 0 Hz (DC)
        

        intermediate_avg = 10 * np.log10(intermediate_avg + 1e-5)
        intermediate_avg *= gain # TODO: Test without gain using subtraction dark frame method

        # Remove Peaks
        window_size = 32
        step_size = 2
        num_points = len(intermediate_avg)
        
        for start in range(0, num_points - window_size + 1, step_size):

            # Define the end of the current window
            end = start + window_size

            intermediate_avg_slice = intermediate_avg[start:end]

            med_slice = np.median(intermediate_avg_slice)
            threshold = 1.1 * med_slice  # Set threshold as 10% of the maximum magnitude

            #mean_spec = np.median(Pxx_dB)
            #for i in range(10):
            peaks, properties = find_peaks(intermediate_avg_slice, height=threshold, prominence=0.3)
            #print(peaks)
            intermediate_avg_slice[peaks] = med_slice
                
            # Put the modified slice back into the original spectrum
            intermediate_avg[start:end] = intermediate_avg_slice

        intermediate_avg = median_filter(intermediate_avg, size=3)


        cumulate_buffer.append(intermediate_avg)

        # Compute the cumulative average over the cumulative buffer
        if len(cumulate_buffer) > 0:
            avg_spectrum = np.mean(cumulate_buffer, axis=0)
        else:
            avg_spectrum = intermediate_avg
            
        print(f"y_data shape: {avg_spectrum.shape}")
            
            
        match shared_data['dark_frame_state']:
            case "Start Record":
                dark_frame_counter = 0
                dark_frame = None
                shared_data['dark_frame_state'] = "Record"

            case "Record":
                dark_frame_counter += 1
                shared_data['dark_frame_status'] = f"Recording Dark Frame: {dark_frame_counter} / {num_cumulative_avg}"
                print(f"Recording Dark Frame: {dark_frame_counter} / {num_cumulative_avg}")

                if dark_frame_counter == num_cumulative_avg:
                    dark_frame = avg_spectrum
                    shared_data['dark_frame_status'] = "Dark Frame Recording Complete"
                    shared_data['dark_frame_state'] = "Complete"

            case "Complete":
                avg_spectrum = avg_spectrum - dark_frame  # Subtraction dark frame
                    # avg_spectrum = avg_spectrum / dark_frame  # Division dark frame

        # Send the result to the plotting queue
        try:
            plot_queue.put(avg_spectrum, block=False)
        except queue.Full:
            print("dropping plot")
            pass
            
            
# Process to start the asynchronous reading
def sdr_async_read():
    sdr.read_samples_async(callback)
   
# Main process to manage SDR and settings updates
def sdr_process():
    global sdr
    async_thread = threading.Thread(target=sdr_async_read, daemon=True)
    async_thread.start()
    async_started = True
    
    while True:
        # If async reading hasn't started, start it in a separate thread
        if not async_started:
            async_thread = threading.Thread(target=sdr_async_read, daemon=True)
            async_thread.start()
            async_started = True
            shared_data['process_settings_state'] = "update_cumulate_buffer"

        # Check if an update is requested
        if shared_data['update_rtlsdr_settings_state'] == "Update":
            print("Cancelling Read Async and updating settings...")
            sdr.cancel_read_async()  # Stop async reading
            async_thread.join()
            async_started = False  # Reset flag
            clear_queue(sample_queue)
            clear_queue(matrix_sample_queue)
            clear_queue(plot_queue)

            # Update SDR settings
            sdr.close()
            sdr = RtlSdr()
            sdr.sample_rate = shared_data['sdr_sample_rate']
            sdr.center_freq = 1420.405e6
            sdr.gain = 48
            sdr.bias_tee = True

            shared_data['update_rtlsdr_settings_state'] = "Nothing"
            
        time.sleep(1)  # Check for updates every second


def clear_queue(q):
    """Function to empty a multiprocessing.Queue."""
    try:
        while not q.empty():
            q.get_nowait()  # Non-blocking dequeue
    except mp.queues.Empty:
        pass  # Handle the case where the queue is empty


# Start the SDR reading process
# sdr_proc = mp.Process(target=sdr_process, daemon=True)
# sdr_proc.start()

sdr_thread = threading.Thread(target=sdr_process, daemon=True)
sdr_thread.start()

# Process into sample matrix
process_sample_matrix_proc = mp.Process(target=process_sample_matrix, args=(sample_queue,), daemon=True)
process_sample_matrix_proc.start()

# Start the sample processing process
processing_proc = mp.Process(target=process_samples, args=(matrix_sample_queue,), daemon=True)
processing_proc.start()



# Serve the HTML page
@app.route('/')
def index():
    integration_minutes = shared_data['integration_minutes']
    return render_template('index.html', integration_minutes=integration_minutes)

# WebSocket handler to send random data
@socketio.on('connect')
def handle_connect():
    print("Client connected")
    def generate_data():
        while True:

            try:
                # Non-blocking attempt to get data
                y_data = plot_queue.get_nowait()
                x_data = np.linspace(-shared_data['sdr_sample_rate'] / (2 * decimation_factor), shared_data['sdr_sample_rate'] / (2 * decimation_factor), nperseg)
                socketio.emit('spectrum_data', {'x_data': x_data.tolist(), 'y_data': y_data.tolist(), 'dark_frame_status': shared_data['dark_frame_status']}) 
            except queue.Empty:
                # No data available, sleep briefly to yield control back to the event loop
                socketio.sleep(0.1)  # Yield control for 100 milliseconds
                continue
                
               
      
    socketio.start_background_task(generate_data)

# WebSocket handler for button click event
@socketio.on('button_clicked')
def handle_button_click(message):
    
    if message['command'].strip() == "record_dark_frame_clicked":
        shared_data['dark_frame_state'] = "Start Record"
    elif message['command'].strip() == "clear_dark_frame_clicked":
        shared_data['dark_frame_state'] = "Initial"

    #print("Button clicked with message:", message['command'])
    emit('server_response', {'response': 'Button click received!'})
    
# WebSocket handler for text input submission
@socketio.on('buffer_settings')
def handle_input_text(message):
    try:
        # Check if 'integration_minutes' is in the message and process it
        if 'integration_minutes' in message:
            integration_minutes = float(message.get('integration_minutes'))
            shared_data['integration_minutes'] = integration_minutes
            print(f"User input (integration_minutes) received: {integration_minutes}")
            
        shared_data['process_settings_state'] = "update_cumulate_buffer"
        
        # Respond back to the client
        emit('server_response', {'response': 'Input received and processed'})

    except (ValueError, TypeError) as e:
        print(f"Invalid number input: {message.get('number')} - {e}")
        emit('server_response', {'response': 'Invalid number input'})


@socketio.on('rtlsdr_settings')
def handle_core_settings(message):
    try:
        if 'rtlsdr_sample_rate' in message:
            sample_rate = float(message.get('rtlsdr_sample_rate'))
            print(f"User input (sample_rate) received: {sample_rate}")

        shared_data['sdr_sample_rate'] = sample_rate * 10**6
        shared_data['update_rtlsdr_settings_state'] = "Update"
        
        # Respond back to the client
        emit('server_response', {'response': 'Input received and processed'})

    except (ValueError, TypeError) as e:
        print(f"Invalid number input: {message.get('number')} - {e}")
        emit('server_response', {'response': 'Invalid number input'})




if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)