import argparse
import logging
import wave
from datetime import datetime, timedelta
from glob import glob
from os import remove, makedirs
from os.path import exists, join, basename
from typing import List, NamedTuple

from numpy import ndarray, zeros
from scipy.io import wavfile

from Utils import WAV_TIMESTAMP_FORMAT, Args

metadata_type = NamedTuple("WavMetadata", (("path", str), ("start", datetime), ("duration", float), ("end", datetime)))


def parse_metadata(f: str) -> metadata_type:
    start: datetime = parse_start_time(f)
    duration: float = get_wav_duration(f)
    if duration is not None:
        return metadata_type(start=start, duration=duration, path=f, end=start + timedelta(seconds=duration))


def parse_start_time(filename: str) -> datetime:
    return datetime.strptime(basename(filename).split("_")[0], WAV_TIMESTAMP_FORMAT)


def get_wav_duration(filepath: str) -> float:
    """Return duration (seconds) of a .wav file."""
    try:
        with wave.open(filepath, 'rb') as wf:
            return wf.getnframes() / float(wf.getframerate())
    except EOFError:
        logging.error(f"File corrupted: {filepath}")
        return None


def batch_split(files: List[str], segment_minutes: int, outdir: str) -> None:
    """
    Split all wav files into consecutive segments of segment_minutes.
    """

    if len(files) == 0:
        logging.error("No files provided to split.")
        raise Exception("No files provided.")

    if exists(outdir):
        logging.debug("Out dir exists, cleaning.")
        for f in glob(join(outdir, "*")):
            remove(f)
    else:
        logging.debug("Out dir not exists, creating.")
        makedirs(outdir)

    # List > tuple here as items will be removed
    null_value: datetime = datetime.fromordinal(1)
    file_info_list: List[metadata_type] = sorted((parse_metadata(file) for file in files), key=lambda a: getattr(a, "start", null_value))
    for i in range(len(file_info_list)-1, -1, -1):  # work backwards to be able to remove items as it iterates
        if file_info_list[i] is None:
            logging.debug(f"Removing null file data {i}")
            file_info_list.pop(i)
    start_time: datetime = file_info_list[0].start
    end_time: datetime = file_info_list[-1].end
    total_duration: timedelta = end_time - start_time
    current_idx: int = 0
    segment_delta: timedelta = timedelta(minutes=segment_minutes)
    current_end: datetime = start_time + segment_delta
    sample_rate: int = -1
    segment_data: ndarray[int]
    logging.debug(f"Starting split of {len(file_info_list)} files with time delta of {segment_minutes} minutes.")
    while start_time < end_time:
        if current_idx == len(file_info_list):
            if sample_rate != -1:
                wavfile.write(join(outdir, f"{start_time.strftime(WAV_TIMESTAMP_FORMAT)}.wav"), sample_rate, segment_data)
                sample_rate = -1
                del segment_data
                logging.info(f"File with start time {start_time.strftime(WAV_TIMESTAMP_FORMAT)} created. {round(100 - (100 * (end_time - start_time) / total_duration), 2)}% complete.")

            current_idx = 0
            current_end += segment_delta
            start_time += segment_delta

        f = file_info_list[current_idx]

        if f.end < start_time:
            # file ends before we start, can be discarded
            file_info_list.pop(current_idx)
            logging.debug(f"Removing file from list. {len(file_info_list)} files remain.")
            if len(file_info_list) == 0:
                break
            continue

        if f.start > current_end:
            # if file starts after we end reached the end of the files for this segment
            current_idx = len(file_info_list)
            continue

        # add/build to file in this duration
        sr: int
        data: ndarray
        sr, data = wavfile.read(f.path)
        if sample_rate != sr and sample_rate != -1:
            logging.error("Sample rates do not match.")
            raise Exception("Sample rates do not match.")
        elif sample_rate == -1:
            sample_rate = sr
            segment_data = zeros(int(sample_rate * 60 * segment_minutes), data.dtype)

        # get file start and stop relative to the segment
        f_start: int = int((max(f.start, start_time) - f.start).total_seconds() * sample_rate)
        f_end: int = int((min(f.end, current_end) - f.start).total_seconds() * sample_rate)

        # get segment start and stop relative to the file
        seg_start: int = int((max(f.start, start_time) - start_time).total_seconds() * sample_rate)
        seg_stop: int = int((min(f.end, current_end) - start_time).total_seconds() * sample_rate)

        segment_data[seg_start:seg_stop] = data[f_start:f_end]

        current_idx += 1

    logging.info("Completed splitting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs the splitter then the analysis on the inputs")

    parser.add_argument("-i", "--input-dir", type=str, required=True, help="Exact filepath to split audio dir")

    parser.add_argument("-ol", "--output-level", type=int, default=1,
                        help="Output level: 0 debug, 1 standard, 2 error only", choices=[0, 1, 2])
    parser.add_argument("-sl", "--segment-length", type=int, default=20, help="Length of each segment")

    args: Args = Args()
    parser.parse_args(namespace=args)

    if args.output_level == 0:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.output_level == 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.ERROR)

    SEGMENT_DIR: str = join(args.input_dir, "output_segments")

    makedirs(SEGMENT_DIR, exist_ok=True)

    batch_split(
        glob(join(args.input_dir, "*.wav")),
        args.segment_length,
        SEGMENT_DIR
    )
