import argparse
import logging
from glob import glob
from os import makedirs
from os.path import join

from Processing import hydrophone_processing
from Utils import Args
from WavSplit import batch_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs the splitter then the analysis on the inputs")

    parser.add_argument("-i", "--input-dir", type=str, required=True, help="Exact filepath to audio dir")
    parser.add_argument("-m", "--mat-file", type=str, required=True, help="Exact filepath to .mat file")
    parser.add_argument("-img", "--img-dir", type=str, required=True, help="Exact filepath to outermost dir with images")

    parser.add_argument("-ol", "--output-level", type=int, default=1, help="Output level: 0 debug, 1 standard, 2 error only", choices=[0, 1, 2])
    parser.add_argument("-lc", "--low-cut", type=float, default=500.0, help="The lowcut applied in Hz")
    parser.add_argument("-hc", "--high-cut", type=float, default=20000.0, help="The highcut applied in Hz")
    parser.add_argument("-np", "--nperseg", type=int, default=2048, help="Length of each segment sent to FFT")
    parser.add_argument("-fs", "--fs-calibration", type=float, default=170.0, help="Calibration level")
    parser.add_argument("-fo", "--filter-order", type=int, default=4, help="Order of the bandpass filter")
    parser.add_argument("-so", "--start-offset", type=int, default=0, help="The number of seconds after the start to start the analysis")
    parser.add_argument("-eo", "--end-offset", type=int, default=0, help="The number of seconds before the end to end the analysis")
    parser.add_argument("-sc", "--sample-count", type=int, default=1000000, help="The number of samples to take during analysis. Values < 0 collect as many as possible.")
    parser.add_argument("-sl", "--segment-length", type=int, default=20, help="Length of each segment")


    args: Args = Args()
    parser.parse_args(namespace=args)

    if args.output_level == 0:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.output_level == 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.ERROR)

    segment_dir: str = join(args.input_dir, "output_segments")
    analysis_dir: str = join(args.input_dir, r"output_charts")

    makedirs(segment_dir, exist_ok=True)
    makedirs(analysis_dir, exist_ok=True)

    batch_split(
        glob(join(args.input_dir, "*.wav")),
        args.segment_length,
        segment_dir
    )

    hydrophone_processing(
        wav_folder=segment_dir,
        mat_path=args.mat_file,
        save_path=analysis_dir,
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
