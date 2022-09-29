"""Convert a folder of data to a common format."""


import argparse
import csv
import itertools
import logging
import os
import pathlib
import shutil
import time
import uuid

import pandas as pd
import numpy as np
import scipy
import scipy.interpolate
import yaml


logger = logging.getLogger(__name__)

# number of points to average over at the end of a stabilisation scan to record as the
# stabilised value
POINTS_TO_AVERAGE = 10


def sort_python_measurement_files(folder):
    """Sort a folder of measurement data by order of measurements.

    This doesn't strictly preserve measurement order but rather groups files by
    measurement type in the experiment order.

    Parameters
    ----------
    folder : pathlib.Path
        Folder of data to sort.

    Returns
    -------
    sorted_files : list
        Sorted folder of data.
    """
    # get the set of unique relative file names excluding extensions
    unique_device_file_names = {f.parts[-1].split(".")[0] for f in folder.iterdir()}

    # loop over this set of unique devices and sort by measurement order
    search_exts = [".vt.tsv", ".div*.tsv", ".liv*.tsv", ".mppt.tsv", ".it.tsv"]
    sorted_list = []
    for device_file_name, ext in itertools.product(
        unique_device_file_names, search_exts
    ):
        # use list and sort here because the can be multiple div and liv files
        sorted_list.extend(sorted(list(folder.glob(f"{device_file_name}{ext}"))))

    return sorted_list


def get_scan_dir_and_rsfwd(voc, ascending, r_diff, ncompliance):
    """Determine the scan direction and forward bias series resistance of a sweep.

    Scan direction is dermined based on the sweep direction and sign of Voc. This may
    fail for truly dark I-V sweeps that don't have a Voc.

    Parameters
    ----------
    voc : float or int
        Open-circuit voltage.
    ascending : bool
        Flag for ascending sweep voltage.
    r_diff : numpy.array
        Differential resistance array.
    ncompliance : list
        Data points not in compliance.
    """
    if not isinstance(voc, str):
        if ascending and voc < 0 or (ascending or voc >= 0) and not ascending:
            scan_dir = "fwd"
            rsfwd_index = 1
        else:
            scan_dir = "rev"
            rsfwd_index = -2
        try:
            rsfwd = r_diff[ncompliance][r_diff[ncompliance] >= 0][rsfwd_index]
        except IndexError:
            # no +ve resistance values found that aren't in compliance
            # probably a dead device that's exhibiting noise
            scan_dir = "NA"
            rsfwd = "nan"
    else:
        scan_dir = "NA"
        rsfwd = "nan"

    return scan_dir, rsfwd


def get_setup_dict(data_folder):
    """Generate dictionary of setup .csv files.

    Parameters
    ----------
    data_folder : pathlib.Path
        Folder containing measurement data.

    Returns
    -------
    setup_dict : dict
        Dictionary of setup info.
    """
    # get dictionary of pixel setup files, with one item per file
    setup_files = list(data_folder.glob("**/IV_pixel_setup_*.csv"))
    setup_dict = {}
    for file in setup_files:
        experiment_timestamp = str(file)[-14:-4]
        pixel_setup = pd.read_csv(
            data_folder.joinpath(f"IV_pixel_setup_{experiment_timestamp}.csv"),
            index_col="user_label",
        )
        setup_dict[experiment_timestamp] = pixel_setup

    return setup_dict


def generate_processed_folder(data_folder, tsv_files, processed_folder):
    """Generate folder containing processed data.

    This is equivalent to the processed data from the python plotter.

    Parameters
    ----------
    data_folder : pathlib.Path
        Folder containing measurement data.
    tsv_files : list
        List of measurement data file paths.
    processed_folder : pathlib.Path
        Folder that will containing processed measurement data.
    experiment_timestamp : int
        Experiment timestamp in s since the epoch.
    """
    processed_folder.mkdir()

    processed_header = [
        [
            "voltage (v)",
            "current (A)",
            "time (s)",
            "status",
            "current_density (mA/cm^2)",
            "power_density (mW/cm^2)",
        ]
    ]

    setup_dict = get_setup_dict(data_folder)

    # set a single file modification time for all files so sorting by modification time
    # later doesn't lead to falsely inferring the files were measured in modification
    # order
    mtime = time.time()

    for file in tsv_files:
        logger.info(file)
        # look up area from pixel setup based on analysing the file name
        file_str = str(file.relative_to(data_folder))
        _slot, _label, _device, _timestamp_ext = file_str.split("_")
        _timestamp, _ext, _ = _timestamp_ext.split(".")
        _pixel = int(_device.replace("device", ""))
        _area_type = "dark_area" if "div" in _ext else "area"
        _pixel_setup = setup_dict[_timestamp][
            setup_dict[_timestamp]["mux_index"] == _pixel
        ]
        _area = _pixel_setup[_pixel_setup["system_label"] == _slot].loc[_label][
            _area_type
        ]

        # load and process raw data
        _data = np.genfromtxt(file, delimiter="\t", skip_header=1)
        if _data.ndim == 1:
            # data only has one row so need to reshape as 2D array
            _data = np.expand_dims(_data, axis=0)
        _voltage = _data[:, 0]
        _current = _data[:, 1]
        _time = _data[:, 2]
        _status = _data[:, 3]
        _current_density = _current * 1000 / _area
        _power_density = _current * 1000 * _voltage / _area

        processed_data = np.column_stack(
            (
                _voltage,
                _current,
                _time,
                _status,
                _current_density,
                _power_density,
            )
        ).tolist()

        # write processed data file
        processed_file = processed_folder.joinpath(f"processed_{file_str}")
        with open(processed_file, "w", newline="\n", encoding="utf-8") as open_file:
            writer = csv.writer(open_file, delimiter="\t")
            writer.writerows(processed_header + processed_data)

        # set universal modification and access times
        os.utime(processed_file, (mtime, mtime))


def load_run_args(path):
    """Load run arguments from a yaml file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the run_args yaml file.

    Returns
    -------
    run_args : dict
        Run arguments dictionary.
    """

    class CustomLoader(yaml.SafeLoader):
        """Subclass safe loader to avoid modifying it inplace."""

    def construct_uuid(loader, node):
        mapping = loader.construct_mapping(node)
        return uuid.UUID(int=mapping["int"])

    CustomLoader.add_constructor(
        "tag:yaml.org,2002:python/object:uuid.UUID", construct_uuid
    )

    with open(path, encoding="utf-8") as open_file:
        run_args = yaml.load(open_file, Loader=CustomLoader)

    return run_args


def dummy_interpolation(anything):
    """Replace interpolation with nan when interpoltion fails."""
    return "nan"


def format_folder(data_folder):
    """Change all the data in a folder to a common format.

    If the folder contains data generated from the Python program, re-format it as if
    it was from the LabVIEW program.

    Parameters
    ----------
    data_folder : pathlib.Path
        Folder containing measurement data.

    Returns
    -------
    formatted_folder : pahtlib.Path
        Folder containing formatted data. This will be a new folder path if the data in
        the initial folder required formatting.
    experiment_timestamps : list of int
        Unix time stamps at experiment start times.
    experiment_title :str
        Experiment title.
    """
    processed_folder = data_folder.joinpath("processed")
    analysis_folder = data_folder.joinpath("analysis")
    # create analysis folder if required
    if analysis_folder.exists() is False:
        analysis_folder.mkdir()

    # generate header
    iv_header = [
        [
            "Time (s)",
            "Set voltage (V)",
            "Meas. voltage (V)",
            "Current (A)",
            "J (mA/cm^2)",
            "P (mW/cm^2)",
            "PCE (%)",
            "Timestamp (s)",
            "Status",
            "R_diff (ohms)",
        ]
    ]

    # figure out if this folder holds data from Python program or LabVIEW program by
    # testing for tsv extension (Python program)
    tsv_files = [f for f in data_folder.iterdir() if str(f).endswith(".tsv")]
    python_prog = len(tsv_files) > 0

    if python_prog:
        logger.info("Data probably created with the Python measurement program.")

        # generate processed folder and files if it doesn't already exist
        if processed_folder.exists() is False:
            logger.info("Generating Processed folder...")
            generate_processed_folder(data_folder, tsv_files, processed_folder)
            logger.info("Processed folder generated!")

        processed_files = list(processed_folder.iterdir())

        # sort files by measurement order to allow calculation of derived parameters
        # from other files, e.g. quasi-ff
        if len(set([os.path.getmtime(f) for f in processed_files])) == 1:
            # date modified info hasn't been preserved so data has probably been copied
            # from somewhere else. Fall back on manual determination.
            processed_files = sort_python_measurement_files(processed_folder)
        else:
            # date modified info is available so use it to infer measurement order
            processed_files.sort(key=os.path.getmtime)

        # get run arguments dictionary of dictionaries
        run_args_files = list(data_folder.glob("**/run_args_*.yaml"))
        run_args_dict = {}
        experiment_timestamps = []
        for file in run_args_files:
            _experiment_timestamp = str(file)[-15:-5]
            run_args = load_run_args(
                data_folder.joinpath(f"run_args_{_experiment_timestamp}.yaml")
            )
            run_args_dict[_experiment_timestamp] = run_args
            experiment_timestamps.append(int(_experiment_timestamp))

        # get device data dictionary of dictionaries
        setup_dict = get_setup_dict(data_folder)

        # infer experiment and device details from paths
        experiment_title = str(data_folder.parts[-1])

        pixels_dict = {}
        logger.info("Formatting Python data files...")
        for file in processed_files:
            logger.info(file)

            try:
                _, position, label, pixel, rest = str(file.parts[-1]).split("_")
            except ValueError:
                # the device label probably wasn't provided
                _, position, pixel, rest = str(file.parts[-1]).split("_")
                label = position
            experiment_timestamp, ext1, ext2 = rest.split(".")
            pixel = pixel.strip("device")

            key = f"{label}_{pixel}"
            # add dict key for new pixel to store derived parameters from other files
            if key not in pixels_dict:
                pixels_dict[key] = {}

            # get columns into same format as LabVIEW output
            data = np.genfromtxt(file, delimiter="\t", skip_header=1)
            if len(np.shape(data)) == 1:
                # if there's only one row in a data file numpy will import it as a 1D
                # array so convert it to 2D
                data = np.array([data])

            # apply special formatting to suns_voc voc file if applicable
            _rel_time = data[:, 2] - data[0, 2]
            try:
                if ("vt" in ext1) and (
                    run_args_dict[experiment_timestamp]["suns_voc"] >= 3
                ):
                    # take first portion of voc dwell as ss-voc measurement
                    mask = np.where(
                        _rel_time <= run_args_dict[experiment_timestamp]["i_dwell"]
                    )
                elif ("vt" in ext1) and (
                    run_args_dict[experiment_timestamp]["suns_voc"] <= -3
                ):
                    # take last portion of voc dwell as ss-voc measurement
                    mask = np.where(
                        _rel_time
                        >= _rel_time[-1]
                        - run_args_dict[experiment_timestamp]["i_dwell"]
                    )
                else:
                    mask = [True] * len(data[:, 0])
            except KeyError:
                # suns_voc key probably isn't available for this version of run_args
                mask = [True] * len(data[:, 0])

            rel_time = _rel_time[mask]
            set_voltage = np.full(len(data[:, 0]), np.NaN)[mask]
            meas_voltage = data[:, 0][mask]
            meas_current = data[:, 1][mask]
            time_data = data[:, 2][mask]
            status = data[:, 3][mask]
            meas_j = data[:, 4][mask]
            meas_p = data[:, 5][mask]

            if div := "div" in ext1:
                intensity = 0
                meas_pce = np.zeros(len(meas_p))
            else:
                intensity = 1
                meas_pce = meas_p / intensity

            # measurements not in compliance
            try:
                ncompliance = [not (int(bin(int(s))[-4])) for s in status]
            except IndexError:
                ncompliance = [True for _ in status]
                logger.warning(
                    "WARNING: Invalid status byte format so can't determine "
                    "measurements in complinace."
                )

            timestamp = int(experiment_timestamp) + time_data[0]

            # calculating the area as I / J can lead to occasional floating point
            # rounding area, which causes errors when grouping by area so use string
            # conversion instead
            area = f"{meas_current[0] / (meas_j[0] / 1000):.4f}"

            liv = "liv" in ext1
            if "vt" in ext1:
                # override r_diff length if suns_voc performed using mask
                r_diff = np.zeros(len(data[:, 0][mask]))
                jsc = 0
                voc = 0
                jvff = 0
                pce = 0
                vmp = 0
                jmp = 0
                quasiff = 0
                pcess_pcejv = 0
                scan_rate = 0
                rsh = 0
                rsvoc = 0
                rsfwd = 0
                time_ss = rel_time[-1]
                scan_dir = "-"

                vss = np.mean(meas_voltage[-POINTS_TO_AVERAGE:])
                jss = 0
                pcess = 0
                quasivoc = vss
                lvext = "voc"
                pixels_dict[key]["quasivoc"] = quasivoc
            elif "mpp" in ext1:
                r_diff = np.zeros(len(data[:, 0]))
                jsc = 0
                voc = 0
                jvff = 0
                pce = 0
                vmp = 0
                jmp = 0
                quasiff = 0
                pcess_pcejv = 0
                scan_rate = 0
                rsh = 0
                rsvoc = 0
                rsfwd = 0
                time_ss = rel_time[-1]
                scan_dir = "-"

                vss = np.mean(meas_voltage[-POINTS_TO_AVERAGE:])
                jss = np.mean(meas_j[-POINTS_TO_AVERAGE:])
                pcess = np.absolute(np.mean(meas_pce[-POINTS_TO_AVERAGE:]))
                quasipce = pcess
                lvext = "mpp"
                pixels_dict[key]["quasipce"] = quasipce
                pcess_pcejv = pixels_dict[key]["quasipce"] / pixels_dict[key]["ivpce"]
            elif "it" in ext1:
                r_diff = np.zeros(len(data[:, 0]))
                jsc = 0
                voc = 0
                jvff = 0
                pce = 0
                vmp = 0
                jmp = 0
                quasiff = 0
                pcess_pcejv = 0
                scan_rate = 0
                rsh = 0
                rsvoc = 0
                rsfwd = 0
                time_ss = rel_time[-1]
                scan_dir = "-"

                vss = 0
                jss = np.mean(meas_j[-POINTS_TO_AVERAGE:])
                pcess = 0
                try:
                    quasiff = (
                        pixels_dict[key]["quasipce"]
                        * intensity
                        / (np.absolute(pixels_dict[key]["quasivoc"]) * np.absolute(jss))
                    )
                except KeyError:
                    logger.warning(
                        "WARNING: There was no corresponding mpp scan so can't "
                        "estimate quasi-ff."
                    )
                    quasiff = 0
                lvext = "jsc"
            elif div or liv:
                lvext = ext1

                r_diff = np.gradient(meas_voltage, meas_current)
                try:
                    f_r_diff = scipy.interpolate.interp1d(
                        meas_voltage[ncompliance],
                        r_diff[ncompliance],
                        kind="linear",
                        bounds_error=False,
                        fill_value=0,
                    )
                except ValueError:
                    f_r_diff = dummy_interpolation

                vss = 0
                jss = 0
                pcess = 0
                quasiff = 0
                pcess_pcejv = 0
                scan_rate = (meas_voltage[-1] - meas_voltage[0]) / rel_time[-1]
                rsh = f_r_diff(0)
                time_ss = 0

                try:
                    f_j = scipy.interpolate.interp1d(
                        meas_voltage[ncompliance],
                        meas_j[ncompliance],
                        kind="linear",
                        bounds_error=False,
                        fill_value=0,
                    )
                except ValueError:
                    f_j = dummy_interpolation

                try:
                    f_v = scipy.interpolate.interp1d(
                        meas_j[ncompliance],
                        meas_voltage[ncompliance],
                        kind="linear",
                        bounds_error=False,
                        fill_value=0,
                    )
                except ValueError:
                    f_v = dummy_interpolation

                dpdv = np.gradient(meas_p, meas_voltage)
                try:
                    f_dpdv = scipy.interpolate.interp1d(
                        dpdv[ncompliance],
                        meas_voltage[ncompliance],
                        kind="linear",
                        bounds_error=False,
                        fill_value=0,
                    )
                except ValueError:
                    f_dpdv = dummy_interpolation

                voc = f_v(0)

                # determine scan direction and forward bias series resistance
                ascending = meas_voltage[0] < meas_voltage[-1]
                scan_dir, rsfwd = get_scan_dir_and_rsfwd(
                    voc, ascending, r_diff, ncompliance
                )

                jsc = f_j(0)
                vmp = f_dpdv(0)
                jmp = f_j(vmp)
                if (
                    (not isinstance(jsc, str))
                    and (not isinstance(vmp, str))
                    and (not isinstance(jmp, str))
                ):
                    pmp = vmp * jmp
                    jvff = pmp / (jsc * voc)
                    pce = np.absolute(pmp / intensity)
                else:
                    jvff = "nan"
                    pmp = "nan"
                    pce = "nan"
                try:
                    rsvoc = f_r_diff(voc)
                except NameError:
                    rsvoc = "nan"

                if liv and (not isinstance(pce, str)):
                    if ("liv" in ext1) and ("ivpce" not in pixels_dict[key]):
                        # reset stored jv pce if first liv, for PCE_SS/PCE_JV calc
                        pixels_dict[key]["ivpce"] = pce
                    elif pce > pixels_dict[key]["ivpce"]:
                        # update if new pce is higher
                        pixels_dict[key]["ivpce"] = pce
            else:
                raise ValueError(f"Invalid file extension: {ext1}.")

            # generate new path
            new_file_rel = str(file.relative_to(processed_folder)).replace(
                f"{ext1}.{ext2}", lvext
            )
            new_file = analysis_folder.joinpath(new_file_rel)

            # get data into writable format
            try:
                write_data = np.column_stack(
                    (
                        rel_time,
                        set_voltage,
                        meas_voltage,
                        meas_current,
                        meas_j,
                        meas_p,
                        meas_pce,
                        time_data,
                        status,
                        r_diff,
                    )
                ).tolist()
            except ValueError:
                write_data = []
                logger.error(file, len(rel_time), len(r_diff))

            # get variable name and value
            exp_setup_dict = setup_dict[experiment_timestamp]
            variable_names = exp_setup_dict.columns[7:]
            _pixel_setup = exp_setup_dict[exp_setup_dict["mux_index"] == int(pixel)]
            _pixel_setup_sub = _pixel_setup[_pixel_setup["system_label"] == position]
            variable_values = [
                str(_pixel_setup_sub.loc[label][var]) for var in variable_names
            ]
            # replace nan's with dummy string to prevent indexing errors in seaborn
            # countplots
            variable_values = ["_" if var == 'nan' else var for var in variable_values]
            variable_name = ", ".join(variable_names)
            variable_value = ", ".join(variable_values)

            # get metadata
            metadata = [
                ["Jsc (mA/cm^2)", jsc],
                ["PCE (%)", pce],
                ["Voc (V)", voc],
                ["FF", jvff],
                ["V_MPP (V)", vmp],
                ["J_MPP (mA/cm^2)", jmp],
                ["V_SS (V)", vss],
                ["J_SS (V)", jss],
                ["PCE_SS (%)", pcess],
                ["Quasi-FF", quasiff],
                ["PCE_SS / PCE_JV", pcess_pcejv],
                ["Scan rate (V/s)", scan_rate],
                ["R_sh (ohms)", rsh],
                ["R_s_voc (ohms)", rsvoc],
                ["R_s_vfwd (ohms)", rsfwd],
                ["Time_SS (s)", time_ss],
                ["Keithley IDN", "-"],
                ["Label", label],
                ["Variable", variable_name],
                ["Value", variable_value],
                ["Substrate", "-"],
                ["HTM", "-"],
                ["Perovskite", "-"],
                ["ETM", "-"],
                ["Metal", "-"],
                ["Pixel", pixel],
                ["Position", position],
                ["Intensity (# suns)", intensity],
                ["Assumed Eg (eV)", "-"],
                ["Solar sim", "-"],
                ["Area (cm^2)", area],
                ["Timestamp (s)", timestamp],
                ["Scan number", 0],
                ["Scan direction", scan_dir],
                ["NPLC", 0],
                ["Settling delay (s)", 0],
                ["Compliance (A or V)", 0],
                ["Range (A or V)", 0],
                ["Enable display", "-"],
                ["Enable 4-wire", 0],
                ["Enable concurrent", True],
                ["Concurrent measurements", "Volt, Current"],
                ["Autozero mode", 0],
                ["Path", str(new_file)],
                ["Relative path", str(new_file.parts[-1])],
            ]

            # write new data file
            with open(new_file, "w", newline="\n", encoding="utf-8") as open_file:
                writer = csv.writer(open_file, delimiter="\t")
                writer.writerows(iv_header + write_data + metadata)

        logger.info(
            "Formatting complete! Formatted data can be found in: %s.", analysis_folder
        )
    else:
        logger.info("Data probably created with the LabVIEW measurement program.")

        experiment_title = str(data_folder.parts[-1])

        extensions = [".voc", ".liv1", ".liv2", ".mpp", ".jsc", ".div1", ".div2"]
        data_files = [f for f in data_folder.iterdir() if f.suffix in extensions]

        experiment_timestamps = []
        for file in data_files:
            try:
                _, position, label, pixel, rest = str(file.parts[-1]).split("_")
            except ValueError:
                # the device label probably wasn't provided
                _, position, pixel, rest = str(file.parts[-1]).split("_")
                label = position
            experiment_timestamp, extension = rest.split(".")

            if experiment_timestamp not in experiment_timestamps:
                experiment_timestamps.append(experiment_timestamp)

            new_file = analysis_folder.joinpath(file.parts[-1])
            if extension in [".liv1", ".liv2", ".div1", ".div2"]:
                with open(file, "r", encoding="utf-8") as open_file:
                    reader = csv.reader(open_file, delimiter="\t")
                    header = []
                    data_cols = 0
                    data = []
                    footer = []
                    for index, row in enumerate(reader):
                        if index == 0:
                            header = row
                            data_cols = len(row)
                        elif len(row) == data_cols:
                            data.append(row)
                        else:
                            footer.append(row)

                data = np.array(data, dtype=float)
                meas_voltage = data[:, 2]
                r_diff = data[:, -1]
                status = data[:, -2].astype(int)

                # measurements not in compliance
                ncompliance = [not (int(bin(int(s))[-4])) for s in status]

                # get details from footer
                scan_dir_ix = None
                voc = "nan"
                for index, row in enumerate(footer):
                    if "Scan direction" in row:
                        scan_dir_ix = index
                    elif "Voc (V)" in row:
                        voc = float(row[1])

                # determine scan direction
                ascending = meas_voltage[0] < meas_voltage[-1]
                scan_dir, _ = get_scan_dir_and_rsfwd(
                    voc, ascending, r_diff, ncompliance
                )

                if scan_dir_ix is not None:
                    footer[scan_dir_ix][1] = scan_dir

                with open(new_file, "w", encoding="utf-8") as open_file:
                    writer = csv.writer(open_file, delimiter="\t")
                    writer.writerow(header)
                    writer.writerows(data)
                    writer.writerows(footer)
            else:
                shutil.copy2(file, new_file)

    return analysis_folder, experiment_timestamps, experiment_title


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get folder path")
    parser.add_argument(
        "--folder",
        default=str(pathlib.Path.cwd()),
        help="Absolute path to data folder",
    )
    args = parser.parse_args()
    format_folder(pathlib.Path(args.folder))
