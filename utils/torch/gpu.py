import os
import pynvml

def count_gpus_nvidia():

    try:

        pynvml.nvmlInit()

        device_count = pynvml.nvmlDeviceGetCount()

        return device_count

    except Exception as e:

        print(f"Error counting GPUs: {e}")

        return 0

def define_device(client_id:int=0):

    n_gpus = count_gpus_nvidia()

    if n_gpus:

        os.environ["CUDA_VISIBLE_DEVICES"] = f"{client_id % n_gpus}"

    else:

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
