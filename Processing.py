import argparse
import logging
import math
import os
import shutil
from datetime import datetime, timedelta
from glob import glob
from os.path import join, basename, exists
from typing import List, Iterator

import matplotlib.dates as mdates
import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile, loadmat
from scipy.signal import welch, butter, filtfilt

import Graphs
import IceDetector
from Utils import MatFileData, WAV_TIMESTAMP_FORMAT, Args


def butter_bandpass_filter(
        data: NDArray[np.float64],
        lowcut: float,
        highcut: float,
        fs: int,
        order: int
) -> NDArray[np.float64]:
    nyquist: float = 0.5 * fs
    low: float = lowcut / nyquist
    high: float = highcut / nyquist
    a: float
    b: float
    b, a = butter(order, [low, high], btype='band', output='ba')
    return filtfilt(b, a, data)


def transform(dat: dict, img_folder: str, start: datetime, end: datetime) -> MatFileData:
    date: datetime = (datetime.fromordinal(1) + timedelta(days=float(dat["time"])) - timedelta(days=366))
    img: str = None

    if date < start or date > end:
        logging.debug(f".mat file data {date.isoformat()} is out of bounds, skipping.")
        return None

    for img in sorted(os.listdir(img_folder)):  # must be sorted as os does not guarantee any order of files
        if datetime.fromisoformat(img.replace("[", ":").replace(".jpg", "")) >= date:
            return MatFileData(
                timestamp=date,
                windspd=dat["windspd"],
                sigwaveheight=dat["sigwaveheight"],
                wavespectra=dat["wavespectra"],
                file=img,
            )

    logging.error(f"No image found for date {date.isoformat()} falling back to last.")
    return MatFileData(
        timestamp=date,
        windspd=dat["windspd"],
        sigwaveheight=dat["sigwaveheight"],
        wavespectra=dat["wavespectra"],
        file=img,
    )


def name_to_timestamp(name: str) -> datetime:
    return datetime.strptime(name.split("_PIC_")[1][0:16], "%d%b%Y_%H%M%S")


def create_img_list(img_folder: str, out_folder: str, start_time: datetime, end_time: datetime) -> None:
    for file in os.listdir(out_folder):
        path = join(out_folder, file)
        if os.path.isfile(path):
            os.unlink(path)
    for root, dirs, files in os.walk(img_folder):
        for img in files:
            img_time: datetime = None
            try:
                img_time = name_to_timestamp(img)
            except Exception:
                logging.error(f"failed to parse image name {img} to timestamp, skipping.")

            if img_time and start_time <= img_time <= end_time:
                shutil.copy(join(root, img), join(out_folder, f"{img_time.isoformat()}.jpg".replace(":", "[")))


def classify_images(img_path: str, imgs: List[str]) -> List[IceDetector.ImageClassification]:
    logging.info("Classifying images.")
    model: IceDetector.MODEL_SETUP_DATA = IceDetector.setup_model()
    classifications = []
    for i, img in enumerate(imgs):
        classifications.append(IceDetector.predict(model, img_path=join(img_path, img)))
        logging.debug(f"Classified image {i+1}/{len(imgs)}.")
    return classifications


def get_wav_data(
        wav_folder: str,
        files: List[str],
        nperseg: int,
        low_cut: float,
        high_cut: float,
        filter_order: int,
        start: int
) -> Iterator[List[NDArray[np.float64]]]:
    sr: int = None
    data: NDArray[float] = None
    step = nperseg // 2
    left = start

    def mutate_data():
        nonlocal data, sr
        # sample rates are already known to match
        sr_, data = wavfile.read(join(wav_folder, files.pop(0)))
        if sr and sr_ != sr:
            logging.critical("Samples rates must match.")
            raise Exception("Sr rate does not match")
        sr = sr_
        if data.ndim > 1:
            data = data[:, 0]
        # normalize int -> float
        if data.dtype == np.int16:
            data = data / 32768.0
        elif data.dtype == np.int32:
            data = data / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data - 128) / 128.0
        data = data.astype(np.float64)
        data = butter_bandpass_filter(data, low_cut, high_cut, sr, order=filter_order)

    mutate_data()
    while True:
        ret = []
        while left + nperseg <= len(data):
            ret.append(data[left:left + nperseg])
            left += step
        if len(files) == 0:
            yield ret
            return
        if left < len(data):
            data_p: NDArray = np.zeros(nperseg)
            written: int = len(data)-left
            data_p[0:written] = data[left:]
            mutate_data()
            while nperseg-written > len(data) and len(files):
                data_p[written:written+len(data)] = data
                written += len(data)
                mutate_data()
            left = nperseg-written
            data_p[written:] = data[0:left]
            ret.append(data_p)
        else:
            left -= len(data)
            mutate_data()
        yield ret


def hydrophone_processing(
        wav_folder: str,
        mat_path: str,
        save_path: str,
        low_cut: float,
        high_cut: float,
        filter_order: int,
        nperseg: int,
        fs_level_db: float,
        start_offset: int,
        end_offset: int,
        total_samples: int,
        raw_imgs: str
) -> None:
    logging.info("Beginning processing.")

    # setup folders
    os.makedirs(save_path, exist_ok=True)
    img_folder: str = join(save_path, "imgs")
    os.makedirs(img_folder, exist_ok=True)

    # check input data
    wav_files: List[str] = sorted(glob(join(wav_folder, "*.wav")))
    if len(wav_files) == 0:
        logging.critical(f"No .wav files found in {wav_folder}.")
        raise Exception("No .wav files provided.")
    if not exists(mat_path):
        logging.critical(f".mat file not found at location {mat_path}.")
        raise Exception("No .mat file provided.")

    # Calculate buckets
    sr: int
    bucket_size: int = 1

    # calculate start and end times
    sr, _ = wavfile.read(join(wav_folder, wav_files[-1]))
    start: datetime = datetime.strptime(basename(wav_files[0]).split(".")[0], WAV_TIMESTAMP_FORMAT) + timedelta(seconds=start_offset)
    end: datetime = datetime.strptime(basename(wav_files[-1]).split(".")[0], WAV_TIMESTAMP_FORMAT) + timedelta(seconds=len(_) / sr) - timedelta(seconds=end_offset)
    if total_samples > 0:
        # (total_length / (nperseg / 2)) / total_samples
        bucket_size = math.ceil(((end - start).total_seconds() * sr * 2) / (nperseg * total_samples))
    logging.info(f".wav data start {start.isoformat(timespec='seconds')}, end {end.isoformat(timespec='seconds')}.")
    if start > end:
        logging.critical(f"Start is after end. Check start and end offsets.")
        raise Exception("Start after end")

    # build image folder for .mat to use
    if raw_imgs != "":
        create_img_list(raw_imgs, img_folder, start, end)

    # transform .mat file to usable data
    # https://datadryad.org/dataset/doi:10.5061/dryad.jdfn2z3j1#methods
    raw_mat_file_data: List[MatFileData] = sorted(
        [x for x in [transform(item, img_folder, start, end) for item in loadmat(mat_path, simplify_cells=True)["SWIFT"]] if x]
        , key=lambda x: x.timestamp
    )
    if len(raw_mat_file_data) == 0:
        logging.critical("No .mat data found, file may be corrupted or all data is out of bounds.")
        raise Exception("Bad .mat file data")
    logging.info(f".mat data start {raw_mat_file_data[0].timestamp.isoformat(timespec='seconds')}, end {raw_mat_file_data[-1].timestamp.isoformat(timespec='seconds')}, {len(raw_mat_file_data)} .mat samples.")

    # graph vars
    save_file_name: str = f"{basename(wav_files[0]).split('.')[0]}"
    title: str = f"{start.strftime('%b %d %H:%M')} - {end.strftime('%b %d %H:%M')}"

    # variables used for main iteration
    step: float = (nperseg // 2) / sr
    f_khz: NDArray[float] = None
    freq_mask: NDArray[bool] = None
    psd_list: List[NDArray[np.float64]] = []
    mat_file_data_indexes: List[int] = []
    segment_time_list: List[datetime] = []
    func: Iterator[List[NDArray[np.float64]]] = get_wav_data(
        wav_folder,
        wav_files,
        nperseg,
        low_cut,
        high_cut,
        filter_order,
        start_offset * sr
    )
    bucket: List[NDArray[np.float64]] = []
    logging.info("Examining files.")
    for dat in func:
        if len(mat_file_data_indexes) == total_samples:
            break
        for data in dat:
            # get timestamp based on number of samples taken
            seg_count = len(mat_file_data_indexes) * bucket_size + len(bucket)
            timestamp: datetime = start + timedelta(seconds=seg_count * step)

            if len(data) < nperseg:
                logging.critical("Length of data provided is less than nperseg.")
                raise Exception("Length of data provided is less than nperseg")

            f: NDArray[np.float64]
            psd: float
            f, psd = welch(data, fs=sr, nperseg=nperseg, scaling='density')
            psd_db = 10 * np.log10(psd + np.finfo(float).eps) + fs_level_db

            if freq_mask is None:
                freq_mask = (f >= low_cut) & (f <= high_cut)
                f_khz = f[freq_mask] / 1e3

            k: int = 0
            # NOTE, takes the first timestamp where the value is greater than it
            while timestamp > raw_mat_file_data[k].timestamp and k < len(raw_mat_file_data) - 1:
                k += 1

            # flushes bucket and appends to main data
            if bucket_size == len(bucket):
                psd_list.append(np.mean(bucket, axis=0))
                bucket = []
                if len(mat_file_data_indexes) == total_samples:
                    break
            if len(bucket) == 0:
                mat_file_data_indexes.append(k)
                segment_time_list.append(timestamp)
            bucket.append(psd_db[freq_mask])

            logging.debug(f"{len(mat_file_data_indexes)} samples, {len(bucket)} bucket size.")
        logging.info(f"{len(wav_files)} remaining files, current sample count {len(mat_file_data_indexes)}.")

    if len(bucket) > 0:
        psd_list.append(np.mean(bucket, axis=0))

    logging.info(f"Data collection complete. {len(mat_file_data_indexes)} samples. start {segment_time_list[0].isoformat()} end {segment_time_list[-1].isoformat()}")

    psd: NDArray[float] = np.array(psd_list).T  # transpose
    time_array: NDArray[datetime] = np.array(segment_time_list)
    if img_folder != "":
        images_array: List[IceDetector.ImageClassification] = classify_images(img_folder, [x.file for x in raw_mat_file_data])
    else:
        images_array: List[IceDetector.ImageClassification] = [
            IceDetector.ImageClassification(
                classification=IceDetector.ImageClassification.classification.NA, confidence=1
            ) for x in raw_mat_file_data
        ]
    time_array: NDArray[np.float64] = mdates.date2num(time_array)

    logging.info("Generating graphs.")

    Graphs.save_to_csv(save_path, save_file_name, time_array, f_khz, psd, raw_mat_file_data, mat_file_data_indexes, images_array)

    Graphs.freq_time_spectrograph(save_path, save_file_name, title, time_array, f_khz, psd, high_cut)  # 2a

    Graphs.wind_time_scatter(save_path, save_file_name, title, time_array, raw_mat_file_data, mat_file_data_indexes)  # 2b

    Graphs.wave_time_scatter(save_path, save_file_name, title, time_array, raw_mat_file_data, mat_file_data_indexes)  # 2c

    Graphs.energy_freq_plot(save_path, save_file_name, title, raw_mat_file_data, mat_file_data_indexes)  # 4a

    Graphs.noise_freq_plot(save_path, save_file_name, title, f_khz, psd, low_cut)  # 4b

    Graphs.sound_wind_frequency_plot(save_path, save_file_name, title, f_khz, psd, low_cut, raw_mat_file_data, mat_file_data_indexes)  # 6

    Graphs.psd_frequency_plot(save_path, save_file_name, title, f_khz, psd, low_cut, raw_mat_file_data, mat_file_data_indexes)  # B1

    logging.info("Completed generating graphs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs the analysis on the inputs")

    parser.add_argument("-i", "--input-dir", type=str, required=True, help="Exact filepath to audio dir")
    parser.add_argument("-m", "--mat-file", type=str, required=True, help="Exact filepath to .mat file")

    parser.add_argument("-img", "--img-dir", type=str, default="", help="Exact filepath to outermost dir with images")
    parser.add_argument("-ol", "--output-level", type=int, default=1, help="Output level: 0 debug, 1 standard, 2 error only", choices=[0, 1, 2])
    parser.add_argument("-lc", "--low-cut", type=float, default=500.0, help="The lowcut applied in Hz")
    parser.add_argument("-hc", "--high-cut", type=float, default=20000.0, help="The highcut applied in Hz")
    parser.add_argument("-np", "--nperseg", type=int, default=2048, help="Length of each segment sent to FFT")
    parser.add_argument("-fs", "--fs-calibration", type=float, default=170.0, help="Calibration level")
    parser.add_argument("-fo", "--filter-order", type=int, default=4, help="Order of the bandpass filter")
    parser.add_argument("-so", "--start-offset", type=int, default=0, help="The number of seconds after the start to start the analysis")
    parser.add_argument("-eo", "--end-offset", type=int, default=0, help="The number of seconds before the end to end the analysis")
    parser.add_argument("-sc", "--sample-count", type=int, default=1000000, help="The number of samples to take during analysis. Values < 0 collect as many as possible.")

    args: Args = Args()
    parser.parse_args(namespace=args)

    if args.output_level == 0:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.output_level == 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.ERROR)

    hydrophone_processing(
        wav_folder=args.input_dir,
        mat_path=args.mat_file,
        save_path=join(args.input_dir, "..", r"output_charts"),
        low_cut=args.low_cut,
        high_cut=args.high_cut,
        nperseg=args.nperseg,
        fs_level_db=args.fs_calibration,
        start_offset=args.start_offset,
        end_offset=args.end_offset,
        filter_order=args.filter_order,
        total_samples=args.sample_count,
        raw_imgs=args.img_dir
    )
