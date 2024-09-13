# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Mlflow PythonModel wrapper helper scripts."""

import logging
import os
import tempfile
import pandas as pd
import base64
import io
import json
import PIL
import re
import requests
import torch
import uuid

from PIL import Image, UnidentifiedImageError
from common_constants import DistributedConstants, SDSettingParameters, SDLiterals

logger = logging.getLogger(__name__)

# Uncomment the following line for mlflow debug mode
# logging.getLogger("mlflow").setLevel(logging.DEBUG)


def create_temp_file(request_body: bytes, parent_dir: str) -> str:
    """Create temporory file, save image and return path to the file.

    :param request_body: Image
    :type request_body: bytes
    :param parent_dir: directory name
    :type parent_dir: str
    :return: Path to the file
    :rtype: str
    """
    with tempfile.NamedTemporaryFile(dir=parent_dir, mode="wb", delete=False) as image_file_fp:
        # image_file_fp.write(request_body)
        img_path = image_file_fp.name + ".png"
        try:
            img = Image.open(io.BytesIO(request_body))
        except UnidentifiedImageError as e:
            logger.error("Invalid image format. Please use base64 encoding for input images.")
            raise e
        img.save(img_path)
        return img_path


def process_image(img: pd.Series) -> pd.Series:
    """If input image is in base64 string format, decode it to bytes. If input image is in url format,
    download it and return bytes.
    https://github.com/mlflow/mlflow/blob/master/examples/flower_classifier/image_pyfunc.py

    :param img: pandas series with image in base64 string format or url.
    :type img: pd.Series
    :return: decoded image in pandas series format.
    :rtype: Pandas Series
    """
    image = img[0]
    if isinstance(image, bytes):
        return img
    elif isinstance(image, str):
        if _is_valid_url(image):
            image = requests.get(image).content
            return pd.Series(image)
        else:
            try:
                return pd.Series(base64.b64decode(image))
            except ValueError:
                raise ValueError(
                    "The provided image string cannot be decoded." "Expected format is base64 string or url string."
                )
    else:
        raise ValueError(
            f"Image received in {type(image)} format which is not supported."
            "Expected format is bytes, base64 string or url string."
        )


def process_video(vid: pd.Series) -> str:
    """If input video is in url format, return the video url.
       This function called for each row in the input data, i.e one video a time.

    :param vid: pandas series with valid video url.
    :type vid: pd.Series
    :return: video link str.
    :rtype: str
    """
    video = vid[0]
    if isinstance(video, str):
        if _is_valid_url(video):
            return video
    raise ValueError("Video received is not in valid format. Expected format is url string.")


def _is_valid_url(text: str) -> bool:
    """check if text is url or base64 string
    :param text: text to validate
    :type text: str
    :return: True if url else false
    :rtype: bool
    """
    regex = (
        "((http|https)://)(www.)?"
        + "[a-zA-Z0-9@:%._\\+~#?&//=]"
        + "{2,256}\\.[a-z]"
        + "{2,6}\\b([-a-zA-Z0-9@:%"
        + "._\\+~#?&//=]*)"
    )
    p = re.compile(regex)

    # If the string is empty
    # return false
    if str is None:
        return False

    # Return if the string
    # matched the ReGex
    if re.search(p, text):
        return True
    else:
        return False


def get_current_device() -> torch.device:
    """Get current cuda device
    :return: current device
    :rtype: torch.device
    """

    # check if GPU is available
    if torch.cuda.is_available():
        if os.environ.get(DistributedConstants.LOCAL_RANK) is None:
            msg = "LOCAL_RANK parameter is missing from environment variables."
            logger.warning(f"{msg}")

        # get the current device index
        device_idx = int(os.environ.get(DistributedConstants.LOCAL_RANK, "0"))
        return torch.device(type="cuda", index=device_idx)
    else:
        return torch.device(type="cpu")


def save_image(output_folder: str, img: PIL.Image.Image, format: str) -> str:
    """
    Save image in a folder designated for batch output and return image file path.

    :param output_folder: directory path where we need to save files
    :type output_folder: str
    :param img: image object
    :type img: PIL.Image.Image
    :param format: format to save image
    :type format: str
    :return: file name of image.
    :rtype: str
    """
    filename = f"image_{uuid.uuid4()}.{format.lower()}"
    img.save(os.path.join(output_folder, filename), format=format)
    return filename


def image_to_base64(img: PIL.Image.Image, format: str) -> str:
    """
    Convert image into Base64 encoded string.

    :param img: image object
    :type img: PIL.Image.Image
    :param format: image format
    :type format: str
    :return: base64 encoded string
    :rtype: str
    """
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def get_sd_scheduler_type(scheduler_path: str) -> str:
    """
    Get the scheduler type from the scheduler_config.json file.

    :param scheduler_path: Path to the scheduler directory
    :type scheduler_path: str
    :return: Scheduler type
    :rtype: str
    """
    config_path = os.path.join(scheduler_path, SDSettingParameters.SCHEDULER_CONFIG)
    # Check if the scheduler_config.json file exists at the path
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as file:
                config_data = json.load(file)
            # fetch scheduler class name from the config file
            if SDLiterals.SCHEDULER_CLASS_NAME in config_data:
                return config_data[SDLiterals.SCHEDULER_CLASS_NAME]
            else:
                return SDSettingParameters.DEFAULT_SCHEDULER
        except json.JSONDecodeError:
            return SDSettingParameters.DEFAULT_SCHEDULER
    else:
        # File does not exist, return default
        return SDSettingParameters.DEFAULT_SCHEDULER
