import datetime
import logging
import math
from os.path import join
from sys import maxsize
from typing import Tuple, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from numpy.typing import NDArray

from IceDetector import ImageClassification
from Utils import MatFileData


def save_to_csv(
        save_path: str,
        save_file_name: str,
        time_array: NDArray[np.float64],
        f_khz: NDArray[float],
        psd: NDArray[float],
        raw_mat_file_data: List[MatFileData],
        mat_file_data_indexes: List[int],
        image_array: List[ImageClassification]
):
    def helper_get_index(item: float) -> int:
        i: int
        v: float
        for i, v in enumerate(f_khz):
            if v >= item:
                return i
        logging.error(f"Frequency of {item} kHz is not found.")
        raise Exception(f"Frequency of {item} kHz is not found.")
    logging.info("Saving data to csv.")
    d: NDArray[np.float32] = psd.T.astype(np.float32)
    one_half_khz: int = helper_get_index(1.5)
    seven_half_khz: int = helper_get_index(7.5)
    fifteen_khz: int = helper_get_index(15)
    np.set_printoptions(threshold=maxsize)

    with open(join(save_path, f"{save_file_name}_data.csv"), 'w') as f:
        f.write(f"TIMESTAMP,{f_khz[one_half_khz]} kHz,{f_khz[seven_half_khz]} kHz,{f_khz[fifteen_khz]} kHz,WINDSPEED,WAVEHEIGHT,IMG_CLASSIFICATION,CONFIDENCE\n")
        for i in range(len(time_array)):
            f.write(f"{mdates.num2date(time_array[i]).isoformat()},"
                    f"{d[i][one_half_khz]},"
                    f"{d[i][seven_half_khz]},"
                    f"{d[i][fifteen_khz]},"
                    f"{raw_mat_file_data[mat_file_data_indexes[i]].windspd},"
                    f"{raw_mat_file_data[mat_file_data_indexes[i]].sigwaveheight},"
                    f"{str(image_array[mat_file_data_indexes[i]].classification)},"
                    f"{image_array[mat_file_data_indexes[i]].confidence}\n")

    with open(join(save_path, f"{save_file_name}_psd_data.csv"), 'w') as f:
        f.write(f"TIMESTAMP,{','.join([f'{x} kHz' for x in f_khz])}\n")
        for i in range(len(time_array)):
            f.write(str(time_array[i]))
            f.write(','.join([str(x) for x in d[i]]))
            f.write("\n")


def freq_time_spectrograph(
        save_path: str,
        save_file_name: str,
        title: str,
        time_array: NDArray[np.float64],
        f_khz: NDArray[float],
        psd: NDArray[float],
        high_cut: float
) -> None:
    logging.info("Generating Spectrograph frequency vs time, 1/7.")
    plt.figure(dpi=100, figsize=(4, 3))

    plt.title(title)

    plt.pcolormesh(time_array, f_khz, psd, shading='auto', cmap='viridis')

    plt.colorbar(label="S(f) [dB]")
    plt.clim(0, 80)

    plt.xlabel("Time UTC-0")
    plt.ylabel("f [kHz]")
    h: float = high_cut / 1e3
    plt.yticks([0, h // 2, h])

    ax: plt.gca = plt.gca()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    plt.gca().tick_params(direction="in", which="both")

    plt.tight_layout()
    plt.savefig(join(save_path, f"{save_file_name}_2a.png"))

    plt.close()


def wind_time_scatter(
        save_path: str,
        save_file_name: str,
        title: str,
        time_array: NDArray[np.float64],
        raw_mat_file_data: List[MatFileData],
        mat_file_data_indexes: List[int],
) -> None:
    logging.info("Generating plot Wind speed vs time, 2/7")
    plt.figure(dpi=100, figsize=(4, 3))

    plt.scatter(time_array, [raw_mat_file_data[x].windspd for x in mat_file_data_indexes])

    plt.title(title)

    plt.xlabel("Time UTC-0")
    plt.ylabel("U$_10$[m/s]")

    ax: plt.gca = plt.gca()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    plt.gca().tick_params(direction="in", which="both")

    plt.tight_layout()
    plt.savefig(join(save_path, f"{save_file_name}_2b.png"))

    plt.close()


def wave_time_scatter(
        save_path: str,
        save_file_name: str,
        title: str,
        time_array: NDArray[np.float64],
        raw_mat_file_data: List[MatFileData],
        mat_file_data_indexes: List[int],
) -> None:
    logging.info("Generating plot wave height vs time, 3/7")
    plt.figure(dpi=100, figsize=(4, 3))
    plt.scatter(time_array, [raw_mat_file_data[x].sigwaveheight for x in mat_file_data_indexes])
    plt.title(title)

    plt.xlabel("Time UTC-0")
    plt.ylabel("H$_S$[m]")

    ax: plt.gca = plt.gca()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    plt.gca().tick_params(direction="in", which="both")

    plt.tight_layout()
    plt.savefig(join(save_path, f"{save_file_name}_2c.png"))

    plt.close()


def energy_freq_plot(
        save_path: str,
        save_file_name: str,
        title: str,
        raw_mat_file_data: List[MatFileData],
        mat_file_data_indexes: List[int],
) -> None:
    logging.info("Generating energy frequency plot, 4/7")
    plt.figure(dpi=100, figsize=(4, 4))

    plt.title(title)
    data: List[float] = [raw_mat_file_data[x].wavespectra["energy"] for x in mat_file_data_indexes if not np.all(raw_mat_file_data[x].wavespectra["energy"] == 0)]
    median_energy: NDArray[float]
    if len(data) > 0:
        median_energy = np.median(data, axis=0)
    else:
        median_energy = np.zeros(len(raw_mat_file_data[0].wavespectra["freq"]))
        logging.error("No unique wave spectra data found. Check .mat file.")
    plt.plot(raw_mat_file_data[0].wavespectra["freq"], median_energy)

    plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=8))
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=8))
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().tick_params(direction="in", which="both")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("f [Hz]")
    plt.ylabel("Wave energy, E(f) [m$^2$/Hz]")
    plt.xlim(10**-2, 10**0)
    plt.ylim(10**-2.5, 10**2)

    plt.tight_layout()
    plt.savefig(join(save_path, f"{save_file_name}_4a.png"))

    plt.close()


def noise_freq_plot(
        save_path: str,
        save_file_name: str,
        title: str,
        f_khz: NDArray[float],
        psd: NDArray[float],
        low_cut: float
) -> None:
    logging.info("Generating noise frequency plot, 5/7")
    plt.figure(dpi=100, figsize=(4, 4))

    plt.title(title)
    median_psd: NDArray[float] = np.median(psd, axis=1)
    plt.plot(f_khz, median_psd)

    plt.xlabel("f [kHz]")
    plt.ylabel("Ambient sound, S(f) [dB re 1µPa$^2$/Hz")
    plt.gca().set_xscale("log")
    plt.gca().set_xlim(low_cut / 1e3, 12)

    plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=8))
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().tick_params(direction="in", which="both")

    plt.tight_layout()
    plt.savefig(join(save_path, f"{save_file_name}_4b.png"))

    plt.close()


def sound_wind_frequency_plot(
        save_path: str,
        save_file_name: str,
        title: str,
        f_khz: NDArray[float],
        psd: NDArray[float],
        low_cut: float,
        raw_mat_file_data: List[MatFileData],
        mat_file_data_indexes: List[int],
) -> None:
    logging.info("Generating wind binned sound frequency plot, 6/7.")

    plt.figure(dpi=100, figsize=(4, 4))
    plt.title(title)

    bin_width: int = 3
    wind_array = np.array([raw_mat_file_data[x].windspd for x in mat_file_data_indexes])
    bins: NDArray[int] = np.arange(0, np.ceil(np.max(wind_array)) + bin_width, bin_width)

    bin_indices: NDArray[int] = np.digitize(wind_array, bins)

    for i in range(len(bins)):
        mask: NDArray[int] = bin_indices == i

        if np.sum(mask) == 0:
            continue

        median_psd = np.median(psd[:, mask], axis=1)

        plt.plot(f_khz, median_psd, label=f"{int(bins[i])} m/s")

    plt.xlabel("f [kHz]")
    plt.ylabel("Ambient sound, S(f) [dB re 1µPa$^2$/Hz")
    plt.gca().set_xscale("log")
    plt.gca().set_xlim(low_cut / 1e3, 12)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=8))
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().tick_params(direction="in", which="both")

    plt.tight_layout()
    plt.savefig(join(save_path, f"{save_file_name}_6.png"))

    plt.close()


def psd_frequency_plot(
        save_path: str,
        save_file_name: str,
        title: str,
        f_khz: NDArray[float],
        psd: NDArray[float],
        low_cut: float,
        raw_mat_file_data: List[MatFileData],
        mat_file_data_indexes: List[int],
) -> None:
    logging.info("Generating wind binned psd freq plot 7/7")

    plt.figure(dpi=100, figsize=(4, 3))
    plt.title(title)

    bin_width: int = 5
    wind_array = np.array([raw_mat_file_data[x].windspd for x in mat_file_data_indexes])
    bins: NDArray[int] = np.arange(0, np.ceil(np.max(wind_array)) + bin_width, bin_width)

    bin_indices: NDArray[int] = np.digitize(wind_array, bins)

    for i in range(len(bins)):
        mask: NDArray[int] = bin_indices == i

        if np.sum(mask) == 0:
            continue

        median_psd: NDArray[float] = np.median(psd[:, mask], axis=1)

        plt.plot(f_khz, median_psd, label=f"{int(bins[i])} m/s")

    plt.xlabel("Frequency [kHz]")
    plt.ylabel("psd [dB/Hz]")
    plt.gca().set_xscale("log")
    plt.gca().set_xlim(low_cut / 1e3, 12)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=8))
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().tick_params(direction="in", which="both")

    plt.grid(True, which="both", linestyle="-", alpha=0.3)

    plt.tight_layout()
    plt.savefig(join(save_path, f"{save_file_name}_b1.png"))
    plt.close()
