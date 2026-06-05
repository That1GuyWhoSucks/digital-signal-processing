from argparse import Namespace
from datetime import datetime
from enum import Enum
from typing import NamedTuple


class ImageClassificationCategories(Enum):
    night = "Night"
    ice = "Ice"
    no_ice = "No Ice"
    NA = "NA"

    def __str__(self):
        return self.value


class Args(Namespace):
    input_dir: str
    mat_file: str
    output_level: int
    segment_length: int
    low_cut: float
    high_cut: float
    nperseg: int
    fs_calibration: float
    filter_order: int
    start_offset: int
    end_offset: int
    sample_count: int
    img_dir: str


WAV_TIMESTAMP_FORMAT: str = "%Y%m%dT%H%M%S"

MatFileData = NamedTuple("mat_file_data", (("timestamp", datetime), ("windspd", float), ("sigwaveheight", float), ("wavespectra", any), ("file", str)))

ImageClassification = NamedTuple("image_classification", (("classification", ImageClassificationCategories), ("confidence", float)))
