"""Generate a log file from a folder of data files."""

import argparse
import csv
import os
import pathlib


def generate_log(folder):
    """Generate a log file for a folder of data.

    Parameters
    ----------
    folder : pathlib.Path
        Folder containing data.

    Returns
    -------
    log_path : pathlib.Path
        New log file path.
    """
    print(f"Generating a log file for folder: {folder}")
    log_path = folder.joinpath("log.log")

    files = [
        p
        for p in folder.iterdir()
        if p.suffix in [".liv1", ".liv2", ".mpp", ".voc", ".jsc", ".div1", ".div2"]
    ]

    # sort by date modified to preserve measurement order
    files.sort(key=os.path.getctime)

    # for each data file, list the log metadata
    log = []
    for i, file in enumerate(files):
        with open(file) as f:
            csv_reader = csv.reader(f, delimiter="\t")
            # each row of metadata is a tab-separated list of length 2
            meta = [row for row in csv_reader if len(row) == 2]
        # store the header from the first iteration only
        if i == 0:
            header = [x[0] for x in meta]
            log.append(header)
        data = [x[1] for x in meta]
        log.append(data)

    # the list of metadata to a tab separated file
    with open(log_path, mode="w", newline="\n") as lf:
        csv_writer = csv.writer(lf, delimiter="\t")
        for entry in log:
            csv_writer.writerow(entry)

    print(f"Generated new log file successfully: {log_path}")
    return log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get folder path")
    parser.add_argument(
        "--folder",
        default=str(pathlib.Path.cwd()),
        help="Absolute path to data folder",
    )
    args = parser.parse_args()
    generate_log(pathlib.Path(args.folder))
