"""Convert a folder of data to a common format."""

import argparse
import csv
import os
import pathlib

import numpy as np
import scipy as sp
import scipy.interpolate


def sort_measurement_files(folder):
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
    unique_device_file_names = set(
        [f.parts[-1].split(".")[0] for f in folder.iterdir()]
    )

    # loop over this set of unique devices and sort by measurement order
    search_exts = [".vt.tsv", ".div*.tsv", ".liv*.tsv", ".mppt.tsv", ".it.tsv"]
    sorted_list = []
    for device_file_name in unique_device_file_names:
        for ext in search_exts:
            sorted_list.extend(sorted(list(folder.glob(f"{device_file_name}{ext}"))))

    return sorted_list


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
    start_time : int
        Unix time stamp at experiment start time.
    username : str
        User name.
    experiment_title :str
        Experiment title.
    """
    processed_folder = data_folder.joinpath("processed")
    analysis_folder = data_folder.joinpath("analysis")

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
        print("This is probably a folder created with the Python measurement program.")
        # create analysis folder if required
        if analysis_folder.exists() is False:
            analysis_folder.mkdir()

        processed_files = [f for f in processed_folder.iterdir()]

        # sort files by measurement order to allows calculation of derived parameters
        # from other files, e.g. quasi-ff
        if len(set([os.path.getmtime(f) for f in processed_files])) == 1:
            # date modified info hasn't been preserved so data has probably been copied
            # from somewhere else. Fall back on manual determination.
            processed_files = sort_measurement_files(processed_folder)
        else:
            # date modified info is available so use it to infer measurement order
            processed_files.sort(key=os.path.getmtime)

        pixels_dict = {}
        for ix, file in enumerate(processed_files):
            experiment_title = str(file.parts[-3])
            username = str(file.parts[-4])
            proc, position, label, pixel, rest = str(file.parts[-1]).split("_")
            start_time, ext1, ext2 = rest.split(".")
            pixel = pixel.strip("device")

            key = f"{label}_{pixel}"
            # add dictionary key for new pixel to store derived parameters from other
            # files
            if key not in pixels_dict:
                pixels_dict[key] = {}

            # get columns into same format as LabVIEW output
            data = np.genfromtxt(file, delimiter="\t", skip_header=1)
            if len(np.shape(data)) == 1:
                # if there's only one row in a data file numpy will import it as a 1D
                # array so convert it to 2D
                data = np.array([data])
            rel_time = data[:, 2] - data[0, 2]
            set_voltage = [np.NaN] * len(data[:, 0])
            meas_voltage = data[:, 0]
            meas_current = data[:, 1]
            meas_j = data[:, 4]
            meas_p = data[:, 5]
            if div := "div" in ext1:
                intensity = 0
                meas_pce = np.zeros(len(meas_p))
            else:
                intensity = 1
                meas_pce = meas_p / intensity
            time = data[:, 2]
            status = data[:, 3]

            ascending = meas_voltage[0] < meas_voltage[-1]

            # measurements not in compliance
            ncompliance = [not (int(bin(int(s))[-4])) for s in status]

            timestamp = int(start_time) + time[0]

            area = meas_current[0] / (meas_j[0] / 1000)

            liv = "liv" in ext1
            if div or liv:
                lvext = ext1

                r_diff = np.gradient(meas_voltage, meas_current)
                try:
                    f_r_diff = sp.interpolate.interp1d(
                        meas_voltage[ncompliance],
                        r_diff[ncompliance],
                        kind="linear",
                        bounds_error=False,
                        fill_value=0,
                    )
                except ValueError:
                    f_r_diff = lambda x: "nan"

                vss = 0
                jss = 0
                pcess = 0
                quasiff = 0
                pcess_pcejv = 0
                scan_rate = (meas_voltage[-1] - meas_voltage[0]) / rel_time[-1]
                rsh = f_r_diff(0)
                time_ss = 0

                try:
                    f_j = sp.interpolate.interp1d(
                        meas_voltage[ncompliance],
                        meas_j[ncompliance],
                        kind="linear",
                        bounds_error=False,
                        fill_value=0,
                    )
                except ValueError:
                    f_j = lambda x: "nan"

                try:
                    f_v = sp.interpolate.interp1d(
                        meas_j[ncompliance],
                        meas_voltage[ncompliance],
                        kind="linear",
                        bounds_error=False,
                        fill_value=0,
                    )
                except ValueError:
                    f_v = lambda x: "nan"

                dpdv = np.gradient(meas_p, meas_voltage)
                try:
                    f_dpdv = sp.interpolate.interp1d(
                        dpdv[ncompliance],
                        meas_voltage[ncompliance],
                        kind="linear",
                        bounds_error=False,
                        fill_value=0,
                    )
                except ValueError:
                    f_dpdv = lambda x: "nan"

                voc = f_v(0)
                if type(voc) is not str:
                    if ascending and (voc < 0):
                        scan_dir = "fwd"
                        rsfwd = r_diff[ncompliance][r_diff[ncompliance] >= 0][1]
                    elif not (ascending) and (voc < 0):
                        scan_dir = "rev"
                        rsfwd = r_diff[ncompliance][r_diff[ncompliance] >= 0][-2]
                    elif ascending and (voc >= 0):
                        scan_dir = "rev"
                        rsfwd = r_diff[ncompliance][r_diff[ncompliance] >= 0][-2]
                    elif not (ascending) and (voc >= 0):
                        scan_dir = "fwd"
                        rsfwd = r_diff[ncompliance][r_diff[ncompliance] >= 0][1]
                else:
                    scan_dir = "NA"
                    rsfwd = "nan"

                jsc = f_j(0)
                vmp = f_dpdv(0)
                jmp = f_j(vmp)
                if (
                    (type(jsc) is not str)
                    and (type(vmp) is not str)
                    and (type(jmp) is not str)
                ):
                    ff = vmp * jmp / (jsc * voc)
                    mp = vmp * jmp
                    pce = np.absolute(mp / intensity)
                else:
                    ff = "nan"
                    mp = "nan"
                    pce = "nan"
                rsvoc = f_r_diff(voc)

                if liv and (type(pce) is not str):
                    if ("liv" in ext1) and ("ivpce" not in pixels_dict[key]):
                        # reset stored jv pce if first liv, for PCE_SS/PCE_JV calc
                        pixels_dict[key]["ivpce"] = pce
                    elif pce > pixels_dict[key]["ivpce"]:
                        # update if new pce is higher
                        pixels_dict[key]["ivpce"] = pce
            elif (vt := "vt" in ext1) or (mpp := "mpp" in ext1) or (it := "it" in ext1):
                r_diff = np.zeros(len(data[:, 0]))
                jsc = 0
                voc = 0
                ff = 0
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

                if vt:
                    vss = np.mean(meas_voltage[-10:])
                    jss = 0
                    pcess = 0
                    quasivoc = vss
                    lvext = "voc"
                    pixels_dict[key]["quasivoc"] = quasivoc
                elif mpp:
                    vss = np.mean(meas_voltage[-10:])
                    jss = np.mean(meas_j[-10:])
                    pcess = np.absolute(np.mean(meas_pce[-10:]))
                    quasipce = pcess
                    lvext = "mpp"
                    pixels_dict[key]["quasipce"] = quasipce
                    pcess_pcejv = (
                        pixels_dict[key]["quasipce"] / pixels_dict[key]["ivpce"]
                    )
                elif it:
                    vss = 0
                    jss = np.mean(meas_j[-10:])
                    pcess = 0
                    quasiff = (
                        pixels_dict[key]["quasipce"]
                        * intensity
                        / (np.absolute(pixels_dict[key]["quasivoc"]) * np.absolute(jss))
                    )
                    lvext = "jsc"

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
                        time,
                        status,
                        r_diff,
                    )
                ).tolist()
            except ValueError:
                print(file, len(rel_time), len(r_diff))

            # get metadata
            metadata = [
                ["Jsc (mA/cm^2)", jsc],
                ["PCE (%)", pce],
                ["Voc (V)", voc],
                ["FF", ff],
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
                ["Variable", "-"],
                ["Value", 0],
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
            with open(new_file, "w", newline="\n") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerows(iv_header + write_data + metadata)

        print(
            f"Formatting complete! Formatted data can be found in: {analysis_folder}."
        )
        return analysis_folder, int(start_time), username, experiment_title
    else:
        print("This is probably a folder created with the LabVIEW measurement program.")
        print("Nothing to format.")
        experiment_title = str(data_folder.parts[-1])
        start_time = experiment_title.split("_")[1]
        username = str(data_folder.parts[-2])
        return data_folder, int(start_time), username, experiment_title


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get folder path")
    parser.add_argument(
        "--folder",
        default=str(pathlib.Path.cwd()),
        help="Absolute path to data folder",
    )
    args = parser.parse_args()
    format_folder(pathlib.Path(args.folder))
