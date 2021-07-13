"""Generate a report of solar cell measurement data."""

import os
import pathlib
import time
import warnings

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from matplotlib import axes
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import MSO_ANCHOR
from scipy import constants

from gooey import Gooey, GooeyParser

from format_python_data import format_folder
from log_generator import generate_log

# supress warnings
warnings.filterwarnings("ignore")

# Define a colormap for graded plots
cmap = plt.cm.get_cmap("viridis")


@Gooey(program_name="Data Analysis")
def parse():
    """Parse command line arguments to Gooey GUI"""

    desc = "Analyse solar simulator data and generate a report"
    parser = GooeyParser(description=desc)
    req = parser.add_argument_group(gooey_options={"columns": 1})
    req.add_argument(
        "folder",
        metavar="Folder containing data to be analysed",
        help="Absolute path to the folder containing measurement data",
        widget="DirChooser",
    )
    req.add_argument(
        "fix_ymin_0",
        metavar="Zero y-axis minima",
        help="Fix boxplot y-axis minima to 0",
        widget="Dropdown",
        choices=["yes", "no"],
        default="yes",
    )
    args = parser.parse_args()
    return args


def round_sig_fig(x, sf):
    """
    Rounds a number to the specified number of significant figures.

    Parameters
    ----------
    x : float
        number to round
    sf : float
        number of significant figures

    Returns
    -------
    y : float
        rounded number
    """

    format_str = "%." + str(sf) + "e"
    x_dig, x_ord = map(float, (format_str % x).split("e"))
    return round(x, int(-x_ord) + 1)


def recursive_path_split(filepath):
    """Recursively split filepath into sub-parts delimited by OS file seperator.

    Parameters
    ----------
    path : str
        filepath

    Returns
    -------
    split : tuple
        sub-parts of filepath
    """
    head, tail = os.path.split(filepath)
    if tail == "":
        return (head,)
    return recursive_path_split(head) + (tail,)


def title_image_slide(prs, title):
    """
    Creates a new slide in the presentation (prs) with a formatted title.

    Parameters
    ----------
    prs : presentation object
        pptx presentation object
    title : str
        title of slide

    Returns
    -------
    slide : slide object
        pptx slide object
    """

    # Add a title slide
    title_slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(title_slide_layout)

    # Add text to title and edit its layout
    title_placeholder = slide.shapes.title
    title_placeholder.top = Inches(0)
    title_placeholder.width = Inches(10)
    title_placeholder.height = Inches(0.5)
    title_placeholder.text = title

    # Edit margins within textbox
    text_frame = title_placeholder.text_frame
    text_frame.margin_bottom = Inches(0)
    text_frame.margin_top = Inches(0.1)
    text_frame.vertical_anchor = MSO_ANCHOR.TOP

    # Edit title fontsize and style
    p = text_frame.paragraphs[0]
    run = p.runs[0]
    font = run.font
    font.size = Pt(16)
    font.bold = True

    return slide


def plot_boxplots(df, params, kind, grouping, variable="", i=0, data_slide=None):
    """Create boxplots from the log file.

    Parameters
    ----------
    df : DataFrame
        logfile or group
    params : list of str
        parameters to plot
    kind : str
        kind of paramters plotted
    grouping : str
        how data are grouped
    variable : str
        variable
    i : int
        starting index of boxplot. Useful if carrying on page from previous
        plots.
    data_slide : prs object
        current slide. Useful if carrying on from previous plots.
    """
    plot_labels_dict = {
        "jsc": {"J-V": "Jsc (mA/cm^2)"},
        "voc": {"J-V": "Voc (V)"},
        "pce": {"J-V": "PCE (%)"},
        "vmpp": {"J-V": "Vmp (V)"},
        "jmpp": {"J-V": "Jmp (mA/cm^2)"},
        "jss": {"SSPO": "J_mp_ss (mA/cm^2)", "SSJsc": "J_sc_ss (mA/cm^2)"},
        "pcess": {"SSPO": "PCE_ss (%)"},
        "vss": {"SSPO": "V_mp_ss (V)", "SSVoc": "V_oc_ss (V)"},
        "ff": {"J-V": "FF"},
        "quasiff": {"SSJsc": "Quasi-FF"},
        "rsvfwd": {"J-V": "Rs (ohms)"},
        "rsh": {"J-V": "Rsh (ohms)"},
        "pcesspcejv": {"SSPO": "PCE_ss/PCE_jv"},
    }

    j = 0
    for p in params:
        # create a new slide for every 4 plots
        if (i + j) % 4 == 0:
            ss_or_jv = kind if kind == "J-V" else "Steady-state"
            data_slide = title_image_slide(
                prs,
                f"{variable} {ss_or_jv} Parameters by {grouping}, page {int((i + j) / 4)}",
            )

        # create boxplot
        fig, ax = plt.subplots(
            1, 1, dpi=300, **{"figsize": (A4_width / 2, A4_height / 2)}
        )

        hue = df["scandirection"] if kind == "J-V" else None
        sns.boxplot(
            x=df[grouping],
            y=np.absolute(df[p].astype(float)),
            hue=hue,
            palette="deep",
            linewidth=0.5,
            ax=ax,
            showfliers=False,
        )
        sns.swarmplot(
            x=df[grouping],
            y=np.absolute(df[p].astype(float)),
            hue=hue,
            palette="muted",
            size=3,
            linewidth=0.5,
            edgecolor="gray",
            dodge=True,
            ax=ax,
        )

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], fontsize="small")
        ax.set_xticklabels(
            ax.get_xticklabels(), fontsize="small", rotation=45, ha="right"
        )
        ax.set_xlabel("")
        if p in ["jsc", "voc", "pce", "vmpp", "jmpp", "jss", "pcess", "vss"]:
            if fix_ymin_0:
                ax.set_ylim(0)
        elif p in ["ff", "quasiff"]:
            if fix_ymin_0:
                ax.set_ylim((0, 1))

        ax.set_ylabel(plot_labels_dict[p][kind], fontsize="small")

        fig.tight_layout()

        # save figure and add to powerpoint
        image_png = os.path.join(image_folder, f"boxplot_{p}.png")
        image_svg = os.path.join(image_folder, f"boxplot_{p}.svg")
        fig.savefig(image_png)
        fig.savefig(image_svg)
        data_slide.shapes.add_picture(
            image_png,
            left=lefts[str((i + j) % 4)],
            top=tops[str((i + j) % 4)],
            height=height,
        )
        j += 1

    return i + j, data_slide


def plot_countplots(df, ix, grouping, data_slide, variable=""):
    """Create countplots from the log file.

    Parameters
    ----------
    df : DataFrame
        logfile or group
    ix : int
        figure index
    grouping : str
        how data are grouped
    data_slide: slide
        slide in ppt to add figures to
    variable : str
        variable
    """
    # create count plot
    fig, ax = plt.subplots(1, 1, dpi=300, **{"figsize": (A4_width / 2, A4_height / 2)})
    if grouping == "value":
        ax.set_title(f"{variable}", fontdict={"fontsize": "small"})
    sns.countplot(
        x=df[grouping],
        data=df,
        hue=df["scandirection"],
        linewidth=0.5,
        palette="deep",
        edgecolor="black",
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], fontsize="small")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize="small", rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Number of working pixels", fontsize="small")
    fig.tight_layout()

    # save figure and add to powerpoint
    image_png = os.path.join(image_folder, f"boxchart_yields{ix}.png")
    image_svg = os.path.join(image_folder, f"boxchart_yields{ix}.svg")
    fig.savefig(image_png)
    fig.savefig(image_svg)
    data_slide.shapes.add_picture(
        image_png, left=lefts[str(ix % 4)], top=tops[str(ix % 4)], height=height
    )


def plot_stabilisation(df, title, short_name):
    """Plot stabilisation data.

    Parameters
    ----------
    df : dataFrame
        data to plot
    title : str
        slide title
    short_name : str
        short name for file
    """
    i = 0
    for index, row in df.iterrows():
        # Get label, variable, value, and pixel for title and image path
        label = row["label"]
        variable = row["variable"]
        value = row["value"]
        pixel = row["pixel"]
        vspo = row["vss"]

        # Start a new slide after every 4th figure
        if i % 4 == 0:
            data_slide = title_image_slide(prs, f"{title}, page {int(i / 4)}")

        # Open the data file
        path = row["relativepath"]
        if short_name == "spo":
            cols = (0, 2, 4, 6)
        elif short_name == "sjsc":
            cols = (0, 4)
        elif short_name == "svoc":
            cols = (0, 2)
        s = np.genfromtxt(
            path, delimiter="\t", skip_header=1, skip_footer=num_cols, usecols=cols
        )
        try:
            s = s[~np.isnan(s).any(axis=1)]
        except:
            pass

        try:
            if short_name == "spo":
                fig, ax = plt.subplots(
                    3, 1, sharex=True, figsize=(A4_width / 2, A4_height / 2), dpi=300
                )
                ax1, ax2, ax3 = ax
                fig.subplots_adjust(hspace=0)
                ax1.set_title(
                    f"{label}, pixel {pixel}, {variable}, {value}, vspo = {vspo} V",
                    fontdict={"fontsize": "small"},
                )
                ax1.scatter(
                    s[:, 0], np.absolute(s[:, 2]), color="black", s=5, label="J"
                )
                ax1.set_ylabel("|J| (mA/cm^2)", fontsize="small")
                ax1.set_ylim(0, np.max(np.absolute(s[:, 2])) * 1.1)
                ax3.tick_params(direction="in", top=True, right=True)
                ax2.scatter(
                    s[:, 0],
                    np.absolute(s[:, 3]),
                    color="red",
                    s=5,
                    marker="s",
                    label="pce",
                )
                ax2.set_ylabel("PCE (%)", fontsize="small")
                ax2.set_ylim(0, np.max(np.absolute(s[:, 3])) * 1.1)
                ax2.tick_params(direction="in", top=True, right=True)
                ax3.scatter(
                    s[:, 0],
                    np.absolute(s[:, 1]),
                    color="blue",
                    s=5,
                    marker="s",
                    label="v",
                )
                ax3.set_ylabel("V (V)", fontsize="small")
                ax3.set_ylim(0, np.max(np.absolute(s[:, 1])) * 1.1)
                ax3.set_xlabel("Time (s)", fontsize="small")
                ax3.tick_params(direction="in", top=True, right=True, labelsize="small")
                fig.align_ylabels([ax1, ax2, ax3])
            elif short_name == "sjsc":
                fig = plt.figure(figsize=(A4_width / 2, A4_height / 2), dpi=300)
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_title(
                    f"{label}, pixel {pixel}, {variable}, {value}",
                    fontdict={"fontsize": "small"},
                )
                ax1.scatter(
                    s[:, 0], np.absolute(s[:, 1]), color="black", s=5, label="Jsc"
                )
                ax1.set_ylabel("|Jsc| (mA/cm^2)", fontsize="small")
                ax1.set_ylim(0, np.max(np.absolute(s[:, 1])) * 1.1)
                ax1.set_xlabel("Time (s)", fontsize="small")
                ax1.set_xlim(0)
                ax1.tick_params(direction="in", top=True, right=True, labelsize="small")
                fig.tight_layout()
            elif short_name == "svoc":
                fig = plt.figure(figsize=(A4_width / 2, A4_height / 2), dpi=300)
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_title(
                    f"{label}, pixel {pixel}, {variable}, {value}",
                    fontdict={"fontsize": "small"},
                )
                ax1.scatter(
                    s[:, 0], np.absolute(s[:, 1]), color="black", s=5, label="Voc"
                )
                ax1.set_ylabel("|Voc| (V)", fontsize="small")
                ax1.set_ylim(0, np.max(np.absolute(s[:, 1])) * 1.1)
                ax1.set_xlabel("Time (s)", fontsize="small")
                ax1.set_xlim(0)
                ax1.tick_params(direction="in", top=True, right=True, labelsize="small")
                fig.tight_layout()

            # Format the figure layout, save to file, and add to ppt
            image_png = os.path.join(
                image_folder, f"{short_name}_{label}_{variable}_{value}_{pixel}.png"
            )
            image_svg = os.path.join(
                image_folder, f"{short_name}_{label}_{variable}_{value}_{pixel}.svg"
            )
            fig.savefig(image_png)
            fig.savefig(image_svg)
            data_slide.shapes.add_picture(
                image_png, left=lefts[str(i % 4)], top=tops[str(i % 4)], height=height
            )

            # Close figure
            plt.close(fig)
        except IndexError:
            print("indexerror")
            pass

        i += 1


def plot_spectra(files):
    """Plot illumination specta.

    Parameters
    ----------
    files : list
        list of file paths
    """
    c_div = 1 / len(files)

    data_slide = title_image_slide(prs, "Measured illumination spectra")

    fig, ax = plt.subplots(1, 1, figsize=(A4_width, A4_height), dpi=300)
    for i, f in enumerate(files):
        if os.path.getsize(f) != 0:
            spectrum = np.genfromtxt(f, delimiter="\t")
            ax.plot(spectrum[:, 0], spectrum[:, 1], color=cmap(i * c_div), label=f"{i}")
    ax.set_ylabel("Spectral irradiance (W/cm^2/nm)", fontsize="large")
    ax.set_ylim(0)
    ax.tick_params(direction="in", top=True, right=True, labelsize="large")
    ax.set_xlabel("Wavelength (nm)", fontsize="large")
    ax.set_xlim(350, 1100)
    ax.legend(fontsize="large")

    fig.tight_layout()

    # Format the figure layout, save to file, and add to ppt
    image_png = os.path.join(image_folder, "spectra.png")
    image_svg = os.path.join(image_folder, "spectra.svg")
    fig.savefig(image_png)
    fig.savefig(image_svg)
    data_slide.shapes.add_picture(
        image_png, left=lefts["0"], top=tops["0"], height=height * 2
    )

    # Close figure
    plt.close(fig)


# parse args
args = parse()

if args.fix_ymin_0 == "yes":
    fix_ymin_0 = True
else:
    fix_ymin_0 = False

# format data if from Python program
(analysis_folder, start_time, username, experiment_title,) = format_folder(
    pathlib.Path(args.folder)
)

# generate log file
log_filepath = generate_log(analysis_folder)

# Define folder and file paths
folderpath, log_filename = os.path.split(log_filepath)
folderpath_split = recursive_path_split(folderpath)
if folderpath_split[-1] == "LOGS":
    raise ValueError("the log file must be in the same folder as the jv data!")

# change cwd to same folder as log file
os.chdir(folderpath)

# Create folders for storing files generated during analysis
print("Creating analysis folder...", end="", flush=True)
analysis_folder = os.path.join(folderpath, "Analysis")
image_folder = os.path.join(analysis_folder, "Figures")
if os.path.exists(analysis_folder):
    pass
else:
    os.makedirs(analysis_folder)
if os.path.exists(image_folder):
    pass
else:
    os.makedirs(image_folder)
print("Done")

# Get username, date, and title from folderpath for the ppt title page
exp_date = time.strftime("%A %B %d %Y", time.localtime(start_time))

# Set physical constants
kB = constants.Boltzmann
q = constants.elementary_charge
c = constants.speed_of_light
h = constants.Planck
T = 300

# Read in data from JV log file
print("Loading log file...", end="", flush=True)
data = pd.read_csv(log_filepath, delimiter="\t", header=0)

# format header names so they can be read as attributes
names = []
for name in data.columns:
    ix = name.find("(")
    if ix != -1:
        # ignore everything from the first parenthesis
        name = name[:ix]
    # remove all special characters
    name = name.lower().translate(
        {ord(c): "" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+ "}
    )
    names.append(name)
data.columns = names

num_cols = len(data.columns)
print("Done")

# Create a powerpoint presentation to add figures to.
print("Creating powerpoint file...", end="", flush=True)
prs = Presentation()

# Add title page with experiment title, date, and username.
title_slide_layout = prs.slide_layouts[0]
slide = prs.slides.add_slide(title_slide_layout)
title = slide.shapes.title
subtitle = slide.placeholders[1]
title.text = experiment_title
subtitle.text = f"{exp_date}\n{username}"

# Add blank slide for table of experimental details.
blank_slide_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank_slide_layout)
shapes = slide.shapes
rows = len(data["label"].unique()) + 1
cols = 6
left = Inches(0.15)
top = Inches(0.02)
width = prs.slide_width - Inches(0.25)
height = prs.slide_height - Inches(0.05)
table = shapes.add_table(rows, cols, left, top, width, height).table

# set column widths
table.columns[0].width = Inches(0.8)
table.columns[1].width = Inches(0.6)
table.columns[2].width = Inches(2.2)
table.columns[3].width = Inches(2.2)
table.columns[4].width = Inches(2.2)
table.columns[5].width = Inches(1.7)

# write column headings
table.cell(0, 0).text = "label"
table.cell(0, 1).text = "sub"
table.cell(0, 2).text = "HTM"
table.cell(0, 3).text = "perovskite"
table.cell(0, 4).text = "ETM"
table.cell(0, 5).text = "metal"

# Define dimensions used for adding images to slides
A4_height = 7.5
A4_width = 10
height = prs.slide_height * 0.95 / 2
width = prs.slide_width * 0.95 / 2

# Create dictionaries that define where to put images on slides in the ppt
lefts = {
    "0": Inches(0),
    "1": prs.slide_width - width,
    "2": Inches(0),
    "3": prs.slide_width - width,
}
tops = {
    "0": prs.slide_height * 0.05,
    "1": prs.slide_height * 0.05,
    "2": prs.slide_height - height,
    "3": prs.slide_height - height,
}
print("Done")

# Sort data
print("Sorting and filtering data...", end="", flush=True)

# filter out pixels that yield a non-physical quasi-FF or Jss from analysis
filter_groups = data.groupby(["label", "pixel"])
data = filter_groups.filter(
    lambda x: all((x["quasiff"] >= 0) & (x["quasiff"] <= 1) & (x["jss"] < 50))
)

sorted_data = data.sort_values(["label", "pixel", "pce"], ascending=[True, True, False])

# Fill in label column of device info table in ppt
table_info = sorted_data.drop_duplicates(["label"])
i = 1
for ix, row in table_info.iterrows():
    table.cell(i, 0).text = f"{row.label}"
    table.cell(i, 1).text = f"{row.substrate}"
    table.cell(i, 2).text = f"{row.htm}"
    table.cell(i, 3).text = f"{row.perovskite}"
    table.cell(i, 4).text = f"{row.etm}"
    table.cell(i, 5).text = f"{row.metal}"
    i += 1

# Filter data
filtered_data = sorted_data[
    (sorted_data.intensity > 0)
    & (sorted_data.ff > 0)
    & (sorted_data.ff < 1)
    & (np.absolute(sorted_data.jsc) > 0.01)
]
filtered_data_fwd = filtered_data[filtered_data.scandirection == "fwd"]
filtered_data_rev = filtered_data[filtered_data.scandirection == "rev"]
filtered_data = filtered_data.drop_duplicates(["label", "pixel", "scandirection"])

filtered_data_fwd = filtered_data_fwd.drop_duplicates(["label", "pixel"])
filtered_data_rev = filtered_data_rev.drop_duplicates(["label", "pixel"])

# Drop pixels only working in one scannumber scandirection.
# First get the inner merge of the label and pixel columns, i.e. drop rows
# where label and pixel combination only occurs in one scannumber scandirection.
filtered_data_fwd_t = filtered_data_fwd[["label", "pixel"]].merge(
    filtered_data_rev[["label", "pixel"]], on=["label", "pixel"], how="inner"
)
filtered_data_rev_t = filtered_data_rev[["label", "pixel"]].merge(
    filtered_data_fwd[["label", "pixel"]], on=["label", "pixel"], how="inner"
)

# Then perform inner merge of full filtered data frames with the merged
# label and pixel dataframes to get back all pixel data that work in both
# scannumber directions
filtered_data_fwd = filtered_data_fwd.merge(
    filtered_data_fwd_t, on=["label", "pixel"], how="inner"
)
filtered_data_rev = filtered_data_rev.merge(
    filtered_data_rev_t, on=["label", "pixel"], how="inner"
)

spo_data = data[(np.absolute(sorted_data.vss) > 0) & (np.absolute(sorted_data.jss) > 0)]
sjsc_data = data[
    (sorted_data.vss == 0)
    & (np.absolute(sorted_data.jss) > 0)
    & (sorted_data.quasiff < 0.9)
    & (sorted_data.quasiff > 0.1)
]
svoc_data = data[(np.absolute(sorted_data.vss) > 0) & (sorted_data.jss == 0)]
print("Done")

print("Plotting spectra...", end="", flush=True)
# Get spectrum files
spectrum_files = [f for f in os.listdir(folderpath) if f.endswith("spectrum.txt")]
if len(spectrum_files) > 0:
    spectrum_files.sort()
    plot_spectra(spectrum_files)
print("Done")

print("Plotting boxplots and barcharts...", end="", flush=True)
jv_params = ["jsc", "voc", "ff", "pce", "vmpp", "jmpp", "rsvfwd", "rsh"]
spo_params = ["pcess", "pcesspcejv"]
sjsc_params = ["jss", "quasiff"]
svoc_params = ["vss"]

# create boxplots for jv and ss parameters grouped by label
if not svoc_data.empty:
    i, data_slide = plot_boxplots(svoc_data, svoc_params, "SSVoc", "label")
    plt.close("all")
if not sjsc_data.empty:
    i, data_slide = plot_boxplots(
        sjsc_data, sjsc_params, "SSJsc", "label", i=i, data_slide=data_slide
    )
    plt.close("all")
if not spo_data.empty:
    i, data_slide = plot_boxplots(
        spo_data, spo_params, "SSPO", "label", i=i, data_slide=data_slide
    )
    plt.close("all")
if not filtered_data.empty:
    i, data_slide = plot_boxplots(filtered_data, jv_params, "J-V", "label")
    plt.close("all")

# create boxplots for jv and spo parameters grouped by variable value
grouped_filtered_data = filtered_data.groupby(["variable"])
grouped_spo_data = spo_data.groupby(["variable"])
grouped_sjsc_data = sjsc_data.groupby(["variable"])
grouped_svoc_data = svoc_data.groupby(["variable"])
for name, group in grouped_svoc_data:
    i, data_slide = plot_boxplots(group, svoc_params, "SSVoc", "value", name)
    plt.close("all")
for name, group in grouped_sjsc_data:
    i, data_slide = plot_boxplots(
        group, sjsc_params, "SSJsc", "value", name, i=i, data_slide=data_slide
    )
    plt.close("all")
for name, group in grouped_spo_data:
    i, data_slide = plot_boxplots(
        group, spo_params, "SSPO", "value", name, i=i, data_slide=data_slide
    )
    plt.close("all")
for name, group in grouped_filtered_data:
    i, data_slide = plot_boxplots(group, jv_params, "J-V", "value", name)
    plt.close("all")

# create countplot for yields grouped by label
ix = 0
data_slide = title_image_slide(prs, f"Yields, page {int(ix / 4)}")
plot_countplots(filtered_data, ix, "label", data_slide)

# create countplot for yields grouped by variable value
ix = 1
for name, group in grouped_filtered_data:
    # create new slide if necessary
    if ix % 4 == 0:
        data_slide = title_image_slide(prs, f"Yields, page {int(ix / 4)}")
    plot_countplots(filtered_data, ix, "value", data_slide, name)
    ix += 1

print("Done")

# plot steady-state data
print("Plotting steady-state data...", end="", flush=True)
plot_stabilisation(spo_data, "Steady-state power output", "spo")
plot_stabilisation(sjsc_data, "Steady-state Jsc", "sjsc")
plot_stabilisation(svoc_data, "Steady-state Voc", "svoc")
print("Done")

print("Plotting JV curves...", end="", flush=True)
# Group data by label and sort ready to plot graph of all pixels per substrate
re_sort_data = filtered_data.sort_values(["label", "pixel"], ascending=[True, True])
grouped_by_label = re_sort_data.groupby("label")

# Define a colormap for JV plots
cmap = plt.cm.get_cmap("viridis")

# Create lists of varibales, values, and labels for labelling figures
substrates = re_sort_data.drop_duplicates(["label"])
variables = list(substrates["variable"])
values = list(substrates["value"])
labels = list(substrates["label"])

# Create figures, save images and add them to powerpoint slide
i = 0
for name, group in grouped_by_label:
    # Create a new slide after every four graphs are produced
    if i % 4 == 0:
        data_slide = title_image_slide(
            prs, f"Best JV scans of every working pixel, page {int(i / 4)}"
        )

    # Create figure, axes, y=0 line, and title
    fig = plt.figure(figsize=(A4_width / 2, A4_height / 2), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(0, lw=0.5, c="black")
    ax.axvline(0, lw=0.5, c="black")
    ax.set_title(
        f"{labels[i]}, {variables[i]}, {values[i]}", fontdict={"fontsize": "small"}
    )

    # get parameters for plot formatting
    c_div = 1 / 8
    pixels = list(group["pixel"].astype(int))
    max_group_jsc = np.max(np.absolute(group["jsc"]))
    max_group_jmp = np.max(np.absolute(group["jmpp"]))
    max_group_voc = np.max(np.absolute(group["voc"]))

    # find signs of jsc and voc to determine max and min axis limits
    jsc_signs, jsc_counts = np.unique(np.sign(group["jmpp"]), return_counts=True)
    voc_signs, voc_counts = np.unique(np.sign(group["voc"]), return_counts=True)
    if len(jsc_signs) == 1:
        jsc_sign = jsc_signs[0]
    else:
        ix = np.argmax(jsc_counts)
        jsc_sign = jsc_signs[ix]
    if len(voc_signs) == 1:
        voc_sign = voc_signs[0]
    else:
        ix = np.argmax(voc_counts)
        voc_sign = voc_signs[ix]

    # load data for each pixel and plot on axes
    fwd_j = []
    rev_j = []
    j = 0
    for file, scan_dir in zip(group["relativepath"], group["scandirection"]):
        if scan_dir == "rev":
            data_rev = np.genfromtxt(
                file,
                delimiter="\t",
                skip_header=1,
                skip_footer=num_cols,
                usecols=(2, 4),
            )
            data_rev = data_rev[~np.isnan(data_rev).any(axis=1)]
            ax.plot(
                data_rev[:, 0],
                data_rev[:, 1],
                label=pixels[j],
                c=cmap(pixels[j] * c_div),
                lw=2.0,
            )
            if (jsc_sign > 0) & (voc_sign > 0):
                fwd_j.append(data_rev[-1, 1])
                rev_j.append(data_rev[0, 1])
            elif (jsc_sign > 0) & (voc_sign < 0):
                fwd_j.append(data_rev[0, 1])
                rev_j.append(data_rev[-1, 1])
            elif (jsc_sign < 0) & (voc_sign > 0):
                fwd_j.append(data_rev[-1, 1])
                rev_j.append(data_rev[0, 1])
            elif (jsc_sign < 0) & (voc_sign < 0):
                fwd_j.append(data_rev[0, 1])
                rev_j.append(data_rev[-1, 1])
        elif scan_dir == "fwd":
            data_fwd = np.genfromtxt(
                file,
                delimiter="\t",
                skip_header=1,
                skip_footer=num_cols,
                usecols=(2, 4),
            )
            data_fwd = data_fwd[~np.isnan(data_fwd).any(axis=1)]
            ax.plot(data_fwd[:, 0], data_fwd[:, 1], c=cmap(pixels[j] * c_div), lw=2.0)
            if (jsc_sign > 0) & (voc_sign > 0):
                fwd_j.append(data_fwd[0, 1])
                rev_j.append(data_fwd[-1, 1])
            elif (jsc_sign > 0) & (voc_sign < 0):
                fwd_j.append(data_fwd[-1, 1])
                rev_j.append(data_fwd[0, 1])
            elif (jsc_sign < 0) & (voc_sign > 0):
                fwd_j.append(data_fwd[0, 1])
                rev_j.append(data_fwd[-1, 1])
            elif (jsc_sign < 0) & (voc_sign < 0):
                fwd_j.append(data_fwd[-1, 1])
                rev_j.append(data_fwd[0, 1])

        j += 1

    # Format the axes
    ax.tick_params(direction="in", top=True, right=True, labelsize="small")
    ax.set_xlabel("Applied bias (V)", fontsize="small")
    ax.set_ylabel("J (mA/cm^2)", fontsize="small")
    # if voc_sign > 0:
    #     ax.set_xlim([np.min(data_fwd[:, 0]), max_group_voc + 0.2])
    # else:
    #     ax.set_xlim([-max_group_voc - 0.2, np.max(data_fwd[:, 0])])
    # if jsc_sign > 0:
    #     ax.set_ylim([-np.max(np.absolute(rev_j)), max_group_jsc * 1.2])
    # else:
    #     ax.set_ylim([-max_group_jsc * 1.2, np.max(np.absolute(rev_j))])

    # Adjust plot width to add legend outside plot area
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labs = ax.get_legend_handles_labels()
    lgd = ax.legend(
        handles,
        labs,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="pixel #",
        fontsize="small",
    )

    # Format the figure layout, save to file, and add to ppt
    image_png = os.path.join(image_folder, f"jv_all_{labels[i]}.png")
    image_svg = os.path.join(image_folder, f"jv_all_{labels[i]}.svg")
    fig.savefig(image_png, bbox_extra_artists=(lgd,), bbox_inches="tight")
    fig.savefig(image_svg, bbox_extra_artists=(lgd,), bbox_inches="tight")
    data_slide.shapes.add_picture(
        image_png, left=lefts[str(i % 4)], top=tops[str(i % 4)], height=height
    )

    # Close figure
    plt.close(fig)

    i += 1

# filter dataframe to leave only the best pixel for each variable value
sort_best_pixels = filtered_data.sort_values(
    ["variable", "value", "pce"], ascending=[True, True, False]
)
best_pixels = sort_best_pixels.drop_duplicates(["variable", "value"])

# get parameters for defining position of figures in subplot, attempting to
# make it as square as possible
no_of_subplots = len(best_pixels["path"])
subplot_rows = np.ceil(no_of_subplots ** 0.5)
subplot_cols = np.ceil(no_of_subplots / subplot_rows)

# create lists of varibales and values for labelling figures
variables = list(best_pixels["variable"])
values = list(best_pixels["value"])
labels = list(best_pixels["label"])
jscs = list(best_pixels["jsc"])
jmps = list(np.absolute(best_pixels["jmpp"]))
vocs = list(np.absolute(best_pixels["voc"]))
jsc_signs = list(np.sign(best_pixels["jmpp"]))
voc_signs = list(np.sign(best_pixels["voc"]))

# Loop for iterating through best pixels dataframe and picking out JV data
# files. Each plot contains forward and reverse sweeps, both light and dark.
i = 0
for file, scan_dir in zip(best_pixels["relativepath"], best_pixels["scandirection"]):
    # Create a new slide after every four graphs are produced
    if i % 4 == 0:
        data_slide = title_image_slide(prs, f"Best pixel JVs, page {int(i / 4)}")

    # Create figure, axes, y=0 line, and title
    fig = plt.figure(figsize=(A4_width / 2, A4_height / 2), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(0, lw=0.5, c="black")
    ax.axvline(0, lw=0.5, c="black")
    ax.set_title(
        f"{variables[i]}, {values[i]}, {labels[i]}", fontdict={"fontsize": "small"}
    )

    # Import data for each pixel and plot on axes, ignoring errors. If
    # data in a file can't be plotted just ignore it.
    if scan_dir == "rev":
        JV_light_rev_path = file
        if file.endswith("liv1"):
            JV_light_fwd_path = file.replace("liv1", "liv2")
        elif file.endswith("liv2"):
            JV_light_fwd_path = file.replace("liv2", "liv1")
    elif scan_dir == "fwd":
        JV_light_fwd_path = file
        if file.endswith("liv1"):
            JV_light_rev_path = file.replace("liv1", "liv2")
        elif file.endswith("liv2"):
            JV_light_rev_path = file.replace("liv2", "liv1")

    try:
        JV_light_rev_data = np.genfromtxt(
            JV_light_rev_path,
            delimiter="\t",
            skip_header=1,
            skip_footer=num_cols,
            usecols=(2, 4),
        )
        JV_light_fwd_data = np.genfromtxt(
            JV_light_fwd_path,
            delimiter="\t",
            skip_header=1,
            skip_footer=num_cols,
            usecols=(2, 4),
        )
        JV_dark_rev_data = np.genfromtxt(
            JV_light_rev_path.replace("liv", "div"),
            delimiter="\t",
            skip_header=1,
            skip_footer=num_cols,
            usecols=(2, 4),
        )
        JV_dark_fwd_data = np.genfromtxt(
            JV_light_fwd_path.replace("liv", "div"),
            delimiter="\t",
            skip_header=1,
            skip_footer=num_cols,
            usecols=(2, 4),
        )
    except OSError:
        pass

    JV_light_rev_data = JV_light_rev_data[~np.isnan(JV_light_rev_data).any(axis=1)]
    JV_light_fwd_data = JV_light_fwd_data[~np.isnan(JV_light_fwd_data).any(axis=1)]

    try:
        JV_dark_rev_data = JV_dark_rev_data[~np.isnan(JV_dark_rev_data).any(axis=1)]
        JV_dark_fwd_data = JV_dark_fwd_data[~np.isnan(JV_dark_fwd_data).any(axis=1)]
    except NameError:
        pass

    # plot light J-V curves
    ax.plot(
        JV_light_rev_data[:, 0], JV_light_rev_data[:, 1], label="rev", c="red", lw=2.0
    )
    ax.plot(
        JV_light_fwd_data[:, 0], JV_light_fwd_data[:, 1], label="fwd", c="black", lw=2.0
    )

    # find y-limits for plotting
    fwd_j = []
    rev_j = []
    if (jsc_signs[i] > 0) & (voc_signs[i] > 0):
        fwd_j.append(JV_light_rev_data[-1, 1])
        rev_j.append(JV_light_rev_data[0, 1])
        fwd_j.append(JV_light_fwd_data[0, 1])
        rev_j.append(JV_light_fwd_data[-1, 1])
    elif (jsc_signs[i] > 0) & (voc_signs[i] < 0):
        fwd_j.append(JV_light_rev_data[0, 1])
        rev_j.append(JV_light_rev_data[-1, 1])
        fwd_j.append(JV_light_fwd_data[-1, 1])
        rev_j.append(JV_light_fwd_data[0, 1])
    elif (jsc_signs[i] < 0) & (voc_signs[i] > 0):
        fwd_j.append(JV_light_rev_data[-1, 1])
        rev_j.append(JV_light_rev_data[0, 1])
        fwd_j.append(JV_light_fwd_data[0, 1])
        rev_j.append(JV_light_fwd_data[-1, 1])
    elif (jsc_signs[i] < 0) & (voc_signs[i] < 0):
        fwd_j.append(JV_light_rev_data[0, 1])
        rev_j.append(JV_light_rev_data[-1, 1])
        fwd_j.append(JV_light_fwd_data[-1, 1])
        rev_j.append(JV_light_fwd_data[0, 1])

    # try to plot dark J-V curves
    try:
        ax.plot(
            JV_dark_rev_data[:, 0],
            JV_dark_rev_data[:, 1],
            label="rev",
            c="orange",
            lw=2.0,
        )
        ax.plot(
            JV_dark_fwd_data[:, 0],
            JV_dark_fwd_data[:, 1],
            label="fwd",
            c="blue",
            lw=2.0,
        )
    except NameError:
        pass

    # Format the axes
    ax.tick_params(direction="in", top=True, right=True, labelsize="small")
    ax.set_xlabel("Applied bias (V)", fontsize="small")
    ax.set_ylabel("J (mA/cm^2)", fontsize="small")
    # if voc_signs[i] > 0:
    #     ax.set_xlim([np.min(JV_light_fwd_data[:, 0]), vocs[i] + 0.1])
    # else:
    #     ax.set_xlim([-vocs[i] - 0.1, np.max(JV_light_fwd_data[:, 0])])

    # if jsc_signs[i] > 0:
    #     ax.set_ylim([-np.max(np.absolute(rev_j)), jscs[i] * 1.2])
    # else:
    #     ax.set_ylim([-jscs[i] * 1.2, np.max(np.absolute(rev_j))])

    ax.legend(loc="best")

    # Format the figure layout, save to file, and add to ppt
    image_png = os.path.join(image_folder, f"jv_best_{variables[i]}_{variables[i]}.png")
    image_svg = os.path.join(image_folder, f"jv_best_{variables[i]}_{variables[i]}.svg")
    fig.tight_layout()
    fig.savefig(image_png)
    fig.savefig(image_svg)
    data_slide.shapes.add_picture(
        image_png, left=lefts[str(i % 4)], top=tops[str(i % 4)], height=height
    )

    # Close figure
    plt.close(fig)

    i += 1

all_jvs_sorted = data.sort_values(
    ["label", "intensity", "pixel", "scannumber"], ascending=[True, False, True, True]
)
all_jvs_groups = all_jvs_sorted[
    (all_jvs_sorted.jsc != 0)
    & (all_jvs_sorted.voc != 0)
    & (all_jvs_sorted.pce != 0)
    & (all_jvs_sorted.ff != 0)
].groupby(["label", "pixel"])

# Create figures, save images and add them to powerpoint slide
i = 0
for name, group in all_jvs_groups:
    # Create a new slide after every four graphs are produced
    if i % 4 == 0:
        data_slide = title_image_slide(prs, f"All JV scans, page {int(i / 4)}")

    label = group["label"].unique()[0]
    pixel = group["pixel"].unique()[0]
    variable = group["variable"].unique()[0]
    value = group["value"].unique()[0]

    fig = plt.figure(figsize=(A4_width / 2, A4_height / 2), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(0, lw=0.5, c="black")
    ax.axvline(0, lw=0.5, c="black")
    ax.set_title(
        f"{label}, {variable}, {value}, pixel {pixel}", fontdict={"fontsize": "small"}
    )

    # get parameters for plot formatting
    c_div = 1 / 4
    scans = group["scannumber"].max() + 1
    max_group_jsc = np.max(np.absolute(group["jsc"]))
    max_group_jmp = np.max(np.absolute(group["jmpp"]))
    max_group_voc = np.max(np.absolute(group["voc"]))

    jsc_signs, jsc_counts = np.unique(np.sign(group["jmpp"]), return_counts=True)
    voc_signs, voc_counts = np.unique(np.sign(group["voc"]), return_counts=True)
    if len(jsc_signs) == 1:
        jsc_sign = jsc_signs[0]
    else:
        ix = np.argmax(jsc_counts)
        jsc_sign = jsc_signs[ix]
    if len(voc_signs) == 1:
        voc_sign = voc_signs[0]
    else:
        ix = np.argmax(voc_counts)
        voc_sign = voc_signs[ix]

    # load data for each pixel and plot on axes
    fwd_j = []
    rev_j = []
    j = 0
    for file, scan_dir, scannumber, intensity in zip(
        group["relativepath"],
        group["scandirection"],
        group["scannumber"],
        group["intensity"],
    ):
        if scan_dir == "rev":
            data_rev = np.genfromtxt(
                file,
                delimiter="\t",
                skip_header=1,
                skip_footer=num_cols,
                usecols=(2, 4),
            )
            data_rev = data_rev[~np.isnan(data_rev).any(axis=1)]
            if intensity == 0:
                ax.plot(
                    data_rev[:, 0],
                    data_rev[:, 1],
                    label=f"{scannumber} {scan_dir} dark",
                    c="black",
                    lw=1.5,
                    ls="--",
                )
            else:
                ax.plot(
                    data_rev[:, 0],
                    data_rev[:, 1],
                    label=f"{scannumber} {scan_dir}",
                    c=cmap(scannumber * c_div),
                    lw=1.5,
                    ls="--",
                )
            if (jsc_sign > 0) & (voc_sign > 0):
                fwd_j.append(data_rev[-1, 1])
                rev_j.append(data_rev[0, 1])
            elif (jsc_sign > 0) & (voc_sign < 0):
                fwd_j.append(data_rev[0, 1])
                rev_j.append(data_rev[-1, 1])
            elif (jsc_sign < 0) & (voc_sign > 0):
                fwd_j.append(data_rev[-1, 1])
                rev_j.append(data_rev[0, 1])
            elif (jsc_sign < 0) & (voc_sign < 0):
                fwd_j.append(data_rev[0, 1])
                rev_j.append(data_rev[-1, 1])
        elif scan_dir == "fwd":
            data_fwd = np.genfromtxt(
                file,
                delimiter="\t",
                skip_header=1,
                skip_footer=num_cols,
                usecols=(2, 4),
            )
            data_fwd = data_fwd[~np.isnan(data_fwd).any(axis=1)]
            if intensity == 0:
                ax.plot(
                    data_fwd[:, 0],
                    data_fwd[:, 1],
                    label=f"{scannumber} {scan_dir} dark",
                    c="black",
                    lw=1.5,
                )
            else:
                ax.plot(
                    data_fwd[:, 0],
                    data_fwd[:, 1],
                    label=f"{scannumber} {scan_dir}",
                    c=cmap(scannumber * c_div),
                    lw=1.5,
                )
            if (jsc_sign > 0) & (voc_sign > 0):
                fwd_j.append(data_fwd[0, 1])
                rev_j.append(data_fwd[-1, 1])
            elif (jsc_sign > 0) & (voc_sign < 0):
                fwd_j.append(data_fwd[-1, 1])
                rev_j.append(data_fwd[0, 1])
            elif (jsc_sign < 0) & (voc_sign > 0):
                fwd_j.append(data_fwd[0, 1])
                rev_j.append(data_fwd[-1, 1])
            elif (jsc_sign < 0) & (voc_sign < 0):
                fwd_j.append(data_fwd[-1, 1])
                rev_j.append(data_fwd[0, 1])

        j += 1

    # Format the axes
    ax.tick_params(direction="in", top=True, right=True, labelsize="small")
    ax.set_xlabel("Applied bias (V)", fontsize="small")
    ax.set_ylabel("J (mA/cm^2)", fontsize="small")
    # if voc_sign > 0:
    #     ax.set_xlim([np.min(data_fwd[:, 0]), max_group_voc + 0.2])
    # else:
    #     ax.set_xlim([-max_group_voc - 0.2, np.max(data_fwd[:, 0])])
    # if jsc_sign > 0:
    #     ax.set_ylim([-np.max(np.absolute(rev_j)), max_group_jsc * 1.2])
    # else:
    #     ax.set_ylim([-max_group_jsc * 1.2, np.max(np.absolute(rev_j))])

    # Adjust plot width to add legend outside plot area
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labs = ax.get_legend_handles_labels()
    lgd = ax.legend(
        handles,
        labs,
        loc="upper left",
        title="scannumber #",
        bbox_to_anchor=(1, 1),
        fontsize="small",
    )

    # Format the figure layout, save to file, and add to ppt
    image_png = os.path.join(image_folder, f"jv_all_{label}.png")
    image_svg = os.path.join(image_folder, f"jv_all_{label}.svg")
    fig.savefig(image_png, bbox_extra_artists=(lgd,), bbox_inches="tight")
    fig.savefig(image_svg, bbox_extra_artists=(lgd,), bbox_inches="tight")
    data_slide.shapes.add_picture(
        image_png, left=lefts[str(i % 4)], top=tops[str(i % 4)], height=height
    )

    # Close figure
    plt.close(fig)

    i += 1

print("Done")

# Save powerpoint presentation
print("Saving powerpoint presentation...", end="", flush=True)
prs.save(str(log_filepath).replace(".log", "_summary.pptx"))
print("Done")
