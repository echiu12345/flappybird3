"""Shared CPU and device settings for command-line execution."""

import os

import torch


def configure_runtime(threads=1, cpu_max=False, device_name=None):
    if cpu_max:
        if device_name not in (None, "cpu"):
            raise ValueError("cpu_max requires the CPU device.")
        threads = os.cpu_count() or 1
        device_name = "cpu"
    if threads <= 0:
        raise ValueError("CPU threads must be positive.")
    torch.set_num_threads(threads)
    return threads, device_name
