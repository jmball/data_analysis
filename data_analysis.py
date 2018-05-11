# This script takes the measurement log files and loads them as Pandas
# DataFrames for manipulation and then plotting.

# import argparse
import os
import time
import ntpath

# import matplotlib
# matplotlib.use('WXAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from matplotlib import axes
from pptx import Presentation
from pptx.util import Inches
from scipy import constants

# import gooey
import reportgenlib as rgl
from gooey import Gooey, GooeyParser


# Bind subplot formatting methods in reportgenlib to the matplotlib.axes.Axes
# class.
axes.Axes.set_axes_props = rgl.set_axes_props
axes.Axes.subboxplot = rgl.subboxplot
axes.Axes.subbarchart = rgl.subbarchart

# # get paths to gooey files
# gooey_root = os.path.dirname(gooey.__file__)
# gooey_images = (os.path.join(gooey_root, 'images/report'))
# print(gooey_images)


# # create Gooey parser
# @Gooey(image_dir=gooey_images,
#        program_name='Data Analysis')
@Gooey(program_name='Data Analysis')
def parse():
    """Parse command line arguments to Gooey GUI"""

    desc = "Analyse solar simulator data and generate a report"
    file_help_msg = "Absolute path to the log file located in the same folder as the measurement data"
    parser = GooeyParser(description=desc)
    parser.add_argument(
        "log_filepath",
        metavar='Filepath to the log file',
        help=file_help_msg,
        widget="FileChooser")
    args = parser.parse_args()
    return args


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
    if tail == '':
        return head,
    return recursive_path_split(head) + (tail,)


args = parse()

# Define folder and file paths
log_filepath = args.log_filepath
# log_filepath_split = log_filepath.split('\\')
# log_filename = log_filepath_split[-1]
# folderpath = log_filepath.replace(log_filename, '')

folderpath, log_filename = ntpath.split(log_filepath)
# print(head)
# print(folderpath)
# print(tail == log_filename)

os.chdir(folderpath)

# Create folders for storing files generated during analysis
print('Creating analysis folder...', end='', flush=True)
# analysis_folder = folderpath + 'Analysis\\'
# image_folder = analysis_folder + 'Figures\\'

analysis_folder = ntpath.join(folderpath, 'Analysis')
image_folder = ntpath.join(analysis_folder, 'Figures')
# print(an)
# print(ntpath.join(an, 'Figures'))

if os.path.exists(analysis_folder):
    pass
else:
    os.makedirs(analysis_folder)
if os.path.exists(image_folder):
    pass
else:
    os.makedirs(image_folder)
print('Done')

# Get username, date, and experiment title from folderpath for the title page
# of the powerpoint version of the report.
# folderpath_split = folderpath.split('\\')
folderpath_split = recursive_path_split(folderpath)
print(folderpath_split)
username = folderpath_split[-2]
exp_date = time.strftime('%A %B %d %Y',
                         time.localtime(os.path.getctime(log_filepath)))
experiment_title = folderpath_split[-1]

# Set physical constants
kB = constants.Boltzmann
q = constants.elementary_charge
c = constants.speed_of_light
h = constants.Planck
T = 300

# Read in data from JV log file
print('Loading log file...', end='', flush=True)
data = pd.read_csv(log_filepath, delimiter='\t', header=None)

num_cols = len(data.columns)

if num_cols == 13:
    names = [
        'Jsc', 'PCE', 'Voc', 'FF', 'Vmp', 'Jspo', 'PCEspo', 'PCEspo-PCE',
        'Label', 'Variable', 'Value', 'Pixel', 'File_Path'
    ]
elif num_cols == 14:
    names = [
        'Jsc', 'PCE', 'Voc', 'FF', 'Vmp', 'Jspo', 'PCEspo', 'PCEspo-PCE',
        'Label', 'Variable', 'Value', 'Pixel', 'Intensity', 'File_Path'
    ]
elif num_cols == 15:
    names = [
        'Jsc', 'PCE', 'Voc', 'FF', 'Vmp', 'Jspo', 'PCEspo', 'PCEspo-PCE',
        'Label', 'Variable', 'Value', 'Pixel', 'Intensity', 'Assumed_Eg',
        'File_Path'
    ]
else:
    raise ValueError(
        f'expected 13, 14, or 15 columns in log file but received {num_cols}')

data.columns = names
print('Done')

# Read scan numbers from file paths and add scan number column
# to dataframe
scan_num = []
for path in data['File_Path']:
    scan_i = path.find('scan', len(path) - 12)
    scan_n = path[scan_i:].strip('scan').strip('_')
    scan_n = scan_n.strip('.liv1')
    scan_n = scan_n.strip('.liv2')
    scan_n = scan_n.strip('.div1')
    scan_n = scan_n.strip('.div2')
    scan_n = scan_n.strip('.hold')
    scan_num.append(scan_n)
data['scan_num'] = pd.Series(scan_num, index=data.index)

# Create a powerpoint presentation to add figures to.
print('Creating powerpoint file...', end='', flush=True)
prs = Presentation()

# Add title page with experiment title, date, and username.
title_slide_layout = prs.slide_layouts[0]
slide = prs.slides.add_slide(title_slide_layout)
title = slide.shapes.title
subtitle = slide.placeholders[1]
title.text = experiment_title
subtitle.text = f'{exp_date}\n{username}'

# Add slide with table for manual completion of experimental details.
blank_slide_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank_slide_layout)
shapes = slide.shapes
rows = 17
cols = 6
left = Inches(0.15)
top = Inches(0.02)
width = prs.slide_width - Inches(0.25)
height = prs.slide_height - Inches(0.05)
table = shapes.add_table(rows, cols, left, top, width, height).table

# set column widths
table.columns[0].width = Inches(0.8)
table.columns[1].width = Inches(1.2)
table.columns[2].width = Inches(2.0)
table.columns[3].width = Inches(2.0)
table.columns[4].width = Inches(2.0)
table.columns[5].width = Inches(1.7)

# write column headings
table.cell(0, 0).text = 'Label'
table.cell(0, 1).text = 'Substrate'
table.cell(0, 2).text = 'Bottom contact'
table.cell(0, 3).text = 'Perovskite'
table.cell(0, 4).text = 'Top contact'
table.cell(0, 5).text = 'Top electrode'

# Define dimensions used for adding images to slides
A4_height = 7.5
A4_width = 10
height = prs.slide_height * 0.95 / 2
width = prs.slide_width * 0.95 / 2

# Create dictionaries that define where to put images on slides
# in the powerpoint presentation.
lefts = {
    '0': Inches(0),
    '1': prs.slide_width - width,
    '2': Inches(0),
    '3': prs.slide_width - width
}
tops = {
    '0': prs.slide_height * 0.05,
    '1': prs.slide_height * 0.05,
    '2': prs.slide_height - height,
    '3': prs.slide_height - height
}
print('Done')

# Use the extra analysis function to perform extra analysis and create new
# series for the dataframe
print('Performing additional JV analysis...')
area_lst = []
scan_direction_lst = []
condition_lst = []
jsc_int_lst = []
voc_int_lst = []
ff_int_lst = []
pce_int_lst = []
vmp_int_lst = []
jmp_int_lst = []
rs_grad_lst = []
rsh_grad_lst = []
rel_path_lst = []
for i in range(len(data['File_Path'])):
    filepath = data['File_Path'][i]
    # new_path = file.replace('\\', '\\\\')
    # params = rgl.extra_JV_analysis(new_path)
    params = rgl.extra_JV_analysis(filepath)
    area_lst.append(params[0])
    scan_direction_lst.append(params[1])
    condition_lst.append(params[2])
    jsc_int_lst.append(params[3])
    voc_int_lst.append(params[4])
    ff_int_lst.append(params[5])
    pce_int_lst.append(params[6])
    vmp_int_lst.append(params[7])
    jmp_int_lst.append(params[8])
    rs_grad_lst.append(params[9])
    rsh_grad_lst.append(params[10])
    rel_path_lst.append(params[11])

# Add new series to the dataframe
data['Area'] = pd.Series(area_lst, index=data.index)
data['Scan_direction'] = pd.Series(scan_direction_lst, index=data.index)
data['Condition'] = pd.Series(condition_lst, index=data.index)
data['Jsc_int'] = pd.Series(jsc_int_lst, index=data.index)
data['Voc_int'] = pd.Series(voc_int_lst, index=data.index)
data['FF_int'] = pd.Series(ff_int_lst, index=data.index)
data['PCE_int'] = pd.Series(pce_int_lst, index=data.index)
data['Vmp_int'] = pd.Series(vmp_int_lst, index=data.index)
data['Jmp_int'] = pd.Series(jmp_int_lst, index=data.index)
data['Rs_grad'] = pd.Series(rs_grad_lst, index=data.index)
data['Rsh_grad'] = pd.Series(rsh_grad_lst, index=data.index)
data['Rel_Path'] = pd.Series(rel_path_lst, index=data.index)
print('Done')

# Sort data
print('Sorting and filtering data...', end='', flush=True)
sorted_data = data.sort_values(
    ['Variable', 'Value', 'Label', 'Pixel', 'PCE_int'],
    ascending=[True, True, True, True, False])

# Fill in label column of device info table in ppt
i = 1
for item in sorted(sorted_data['Label'].unique()):
    table.cell(i, 0).text = f'{item}'
    i += 1

# Filter data
filtered_data = sorted_data[(sorted_data.Condition == 'Light')
                            & (sorted_data.FF_int > 0.1) &
                            (sorted_data.FF_int < 0.9) &
                            (np.absolute(sorted_data.Jsc_int) > 0.01)]
filtered_data_HL = sorted_data[(sorted_data.Condition == 'Light')
                               & (sorted_data.FF_int > 0.1) &
                               (sorted_data.FF_int < 0.9) &
                               (np.absolute(sorted_data.Jsc_int) > 0.01) &
                               (sorted_data.Scan_direction == 'HL')]
filtered_data_LH = sorted_data[(sorted_data.Condition == 'Light')
                               & (sorted_data.FF_int > 0.1) &
                               (sorted_data.FF_int < 0.9) &
                               (np.absolute(sorted_data.Jsc_int) > 0.01) &
                               (sorted_data.Scan_direction == 'LH')]
# filtered_data = filtered_data.drop_duplicates(['Label', 'Pixel'])
filtered_data = filtered_data.drop_duplicates(
    ['Label', 'Pixel', 'Scan_direction'])
filtered_data_HL = filtered_data_HL.drop_duplicates(['Label', 'Pixel'])
filtered_data_LH = filtered_data_LH.drop_duplicates(['Label', 'Pixel'])

spo_data = data[(data.Condition == 'SPO')]

# Drop pixels only working in one scan direction.
# First get the inner merge of the Label and Pixel columns, i.e. drop rows
# where label and pixel combination only occurs in one scan direction.
filtered_data_HL_t = filtered_data_HL[['Label', 'Pixel']].merge(
    filtered_data_LH[['Label', 'Pixel']], on=['Label', 'Pixel'], how='inner')
filtered_data_LH_t = filtered_data_LH[['Label', 'Pixel']].merge(
    filtered_data_HL[['Label', 'Pixel']], on=['Label', 'Pixel'], how='inner')

# Then perform inner merge of full filtered data frames with the merged
# label and pixel dataframes to get back all pixel data that work in both
# scan directions
filtered_data_HL = filtered_data_HL.merge(
    filtered_data_HL_t, on=['Label', 'Pixel'], how='inner')
filtered_data_LH = filtered_data_LH.merge(
    filtered_data_LH_t, on=['Label', 'Pixel'], how='inner')

# Calculate proportion of working pixels.
# Create groups of all and working pixels from complete and filtered
# dataframes.
group_var_s = sorted_data.drop_duplicates(['Label', 'Pixel'])
group_var_s = group_var_s.groupby(['Variable'])
group_var_f = filtered_data_HL.drop_duplicates(['Label', 'Pixel'])
group_var_f = group_var_f.groupby(['Variable'])
print('Done')

# # For each variable, if there are any working pixels for that variable
# # calculate the yield for each value of the variable if there are any working
# # pixels for that value. Else add zeros. Also, make a list of names
# # for each variable to use as the x-axis on bar charts.
# print('Calculating yields...', end='', flush=True)
# yields_var = []
# names_yield_var = []
# for var_key in list(group_var_s.groups.keys()):
#     group_val_s = group_var_s.get_group(var_key).groupby(['Value'])
#     if var_key in list(group_var_f.groups.keys()):
#         group_val_f = group_var_f.get_group(var_key).groupby(['Value'])
#         yields_val = []
#         names_yield_val = []
#         for val_key in list(group_val_s.groups.keys()):
#             names_yield_val.append(val_key)
#             if val_key in list(group_val_f.groups.keys()):
#                 yields_val.append(
#                     len(group_val_f.get_group(val_key)) * 100 / len(
#                         group_val_s.get_group(val_key)))
#             else:
#                 yields_val.append(0)
#         yields_var.append(yields_val)
#         names_yield_var.append(names_yield_val)
#     else:
#         yields_var.append([0] * len(group_val_s))
#         names_yield_var.append(list(group_val_s.groups.keys()))

# # For each variable, if there are any working pixels for that variable
# # calculate the yield for each label if there are any working
# # pixels for that label. Else add zeros. Also, make a list of names
# # for each variable to use as the x-axis on bar charts.
# yields_var_lab = []
# names_yield_var_lab = []
# for var_key in list(group_var_s.groups.keys()):
#     group_lab_s = group_var_s.get_group(var_key).groupby(['Label'])
#     if var_key in list(group_var_f.groups.keys()):
#         group_lab_f = group_var_f.get_group(var_key).groupby(['Label'])
#         yields_lab = []
#         names_yield_lab = []
#         for lab_key in list(group_lab_s.groups.keys()):
#             names_yield_lab.append(lab_key)
#             if lab_key in list(group_lab_f.groups.keys()):
#                 yields_lab.append(
#                     len(group_lab_f.get_group(lab_key)) * 100 / 8)
#             else:
#                 yields_lab.append(0)
#         yields_var_lab.append(yields_lab)
#         names_yield_var_lab.append(names_yield_lab)
#     else:
#         yields_var_lab.append([0] * len(group_lab_s))
#         names_yield_var_lab.append(list(group_lab_s.groups.keys()))
# print('Done')

# # To generate box plots the data needs to be grouped first by variable
# grouped_by_var_HL = filtered_data_HL.groupby('Variable')
# grouped_by_var_LH = filtered_data_LH.groupby('Variable')

print('Plotting boxplots and barcharts...', end='', flush=True)


def plot_boxplots(df, params, kind, grouping, variable=''):
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
    """

    for ix, p in enumerate(params):
        # create a new slide for every 4 plots
        if ix % 4 == 0:
            data_slide = rgl.title_image_slide(
                prs, f'{variable} {kind} Parameters by {grouping}, page {int(ix / 4)}')

        # create boxplot
        fig, ax = plt.subplots(1, 1, **{'figsize': (A4_width / 2, A4_height / 2)})
        if kind == 'J-V':
            sns.boxplot(
                x=df[grouping],
                y=np.absolute(df[p]),
                hue=df['Scan_direction'],
                palette='deep',
                linewidth=0.5,
                ax=ax,
                showfliers=False)
            sns.swarmplot(
                x=df[grouping],
                y=np.absolute(df[p]),
                hue=df['Scan_direction'],
                palette='muted',
                size=3,
                linewidth=0.5,
                edgecolor='gray',
                dodge=True,
                ax=ax)
        elif kind == 'SPO':
            sns.boxplot(
                x=df[grouping],
                y=np.absolute(df[p]),
                palette='deep',
                linewidth=0.5,
                ax=ax,
                showfliers=False)
            sns.swarmplot(
                x=df[grouping],
                y=np.absolute(df[p]),
                palette='muted',
                size=3,
                linewidth=0.5,
                edgecolor='gray',
                dodge=True,
                ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], fontsize='xx-small')
        ax.set_xticklabels(
            ax.get_xticklabels(), fontsize='xx-small', rotation=45, ha='right')
        ax.set_xlabel('')
        if p in ['Jsc_int', 'Voc_int', 'PCE_int', 'Vmp_int', 'Jmp_int']:
            ax.set_ylim(0)
            if p == 'Jsc_int':
                ax.set_ylabel('Jsc (mA/cm^2)')
            elif p == 'Voc_int':
                ax.set_ylabel('Voc (V)')
            elif p == 'PCE_int':
                ax.set_ylabel('PCE (%)')
            elif p == 'Vmp_int':
                ax.set_ylabel('Vmp (V)')
            elif p == 'Jmp_int':
                ax.set_ylabel('Jmp (mA/cm^2)')
        elif p == 'FF_int':
            ax.set_ylim((0, 1))
            ax.set_ylabel('FF')
        elif p in ['Rs_grad', 'Rsh_grad']:
            ax.set_yscale('log')
            if p == 'Rs_grad':
                ax.set_ylabel('Rs (ohms)')
            elif p == 'Rsh_grad':
                ax.set_ylabel('Rsh (ohms)')
        elif p in ['Jspo', 'PCEspo', 'PCEspo-PCE']:
            ax.set_ylim(0)
            if p == 'Jspo':
                ax.set_ylabel('Jspo (mA/cm^2)')
            elif p == 'PCEspo':
                ax.set_ylabel('PCEspo (%)')
            elif p == 'PCEspo-PCE':
                ax.set_ylabel('PCEspo/PCE')
        fig.tight_layout()

        # save figure and add to powerpoint
        image_path = ntpath.join(image_folder, f'boxplot_{p}.png')
        fig.savefig(image_path)
        data_slide.shapes.add_picture(
            image_path,
            left=lefts[str(ix % 4)],
            top=tops[str(ix % 4)],
            height=height)


# create boxplots for jv parameters
jv_params = [
    'Jsc_int', 'Voc_int', 'FF_int', 'PCE_int', 'Vmp_int', 'Jmp_int', 'Rs_grad',
    'Rsh_grad'
]
spo_params = ['Jspo', 'PCEspo', 'PCEspo-PCE']

# create boxplots
plot_boxplots(filtered_data, jv_params, 'J-V', 'Label')
plt.close('all')
plot_boxplots(spo_data, spo_params, 'SPO', 'Label')
plt.close('all')

grouped_filtered_data = filtered_data.groupby(['Variable'])
grouped_spo_data = spo_data.groupby(['Variable'])
for name, group in grouped_filtered_data:
    plot_boxplots(group, jv_params, 'J-V', 'Value', name)
    plt.close('all')

for name, group in grouped_spo_data:
    plot_boxplots(group, spo_params, 'SPO', 'Value', name)
    plt.close('all')

# for ix, p in enumerate(jv_params):
#     # create a new slide for every 4 plots
#     if ix % 4 == 0:
#         data_slide = rgl.title_image_slide(
#             prs, f'J-V Parameters, page {int(ix / 4)}')
#
#     # create boxplot
#     fig, ax = plt.subplots(1, 1, **{'figsize': (A4_width / 2, A4_height / 2)})
#     sns.boxplot(
#         x=filtered_data['Label'],
#         y=np.absolute(filtered_data[p]),
#         hue=filtered_data['Scan_direction'],
#         palette='deep',
#         linewidth=0.5,
#         ax=ax,
#         showfliers=False)
#     sns.swarmplot(
#         x=filtered_data['Label'],
#         y=np.absolute(filtered_data[p]),
#         hue=filtered_data['Scan_direction'],
#         palette='muted',
#         size=3,
#         linewidth=0.5,
#         edgecolor='gray',
#         dodge=True,
#         ax=ax)
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles[:2], labels[:2], fontsize='xx-small')
#     ax.set_xticklabels(
#         ax.get_xticklabels(), fontsize='xx-small', rotation=45, ha='right')
#     ax.set_xlabel('')
#     if p in ['Jsc_int', 'Voc_int', 'PCE_int', 'Vmp_int', 'Jmp_int']:
#         ax.set_ylim(0)
#         if p == 'Jsc_int':
#             ax.set_ylabel('Jsc (mA/cm^2)')
#         elif p == 'Voc_int':
#             ax.set_ylabel('Voc (V)')
#         elif p == 'PCE_int':
#             ax.set_ylabel('PCE (%)')
#         elif p == 'Vmp_int':
#             ax.set_ylabel('Vmp (V)')
#         elif p == 'Jmp_int':
#             ax.set_ylabel('Jmp (mA/cm^2)')
#     elif p == 'FF_int':
#         ax.set_ylim((0, 1))
#         ax.set_ylabel('FF')
#     elif p in ['Rs_grad', 'Rsh_grad']:
#         ax.set_yscale('log')
#         if p == 'Rs_grad':
#             ax.set_ylabel('Rs (ohms)')
#         elif p == 'Rsh_grad':
#             ax.set_ylabel('Rsh (ohms)')
#     fig.tight_layout()
#
#     # save figure and add to powerpoint
#     image_path = ntpath.join(image_folder, f'boxplot_{p}.png')
#     fig.savefig(image_path)
#     data_slide.shapes.add_picture(
#         image_path,
#         left=lefts[str(ix % 4)],
#         top=tops[str(ix % 4)],
#         height=height)
#
# # create boxplots for spo parameters
# spo_params = ['Jspo', 'PCEspo', 'PCEspo-PCE']
# for ix, p in enumerate(spo_params):
#     # create a new slide
#     if ix % 4 == 0:
#         data_slide = rgl.title_image_slide(prs, f'SPO Parameters')
#
#     # create boxplot
#     fig, ax = plt.subplots(1, 1, **{'figsize': (A4_width / 2, A4_height / 2)})
#     sns.boxplot(
#         x=spo_data['Label'],
#         y=np.absolute(spo_data[p]),
#         palette='deep',
#         linewidth=0.5,
#         ax=ax,
#         showfliers=False)
#     sns.swarmplot(
#         x=spo_data['Label'],
#         y=np.absolute(spo_data[p]),
#         palette='muted',
#         size=3,
#         linewidth=0.5,
#         edgecolor='gray',
#         dodge=True,
#         ax=ax)
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles[:2], labels[:2], fontsize='xx-small')
#     ax.set_xticklabels(
#         ax.get_xticklabels(), fontsize='xx-small', rotation=45, ha='right')
#     ax.set_xlabel('')
#     if p == 'Jspo':
#         ax.set_ylabel('Jspo (mA/cm^2)')
#     elif p == 'PCEspo':
#         ax.set_ylabel('PCEspo (%)')
#     ax.set_ylim(0)
#     fig.tight_layout()
#
#     # save figure and add to powerpoint
#     image_path = ntpath.join(image_folder, f'boxplot_{p}.png')
#     fig.savefig(image_path)
#     data_slide.shapes.add_picture(
#         image_path, left=lefts[str(ix)], top=tops[str(ix)], height=height)

# create countplot for yields
# create new slide
data_slide = rgl.title_image_slide(prs, f'Yields')

# create countplot
fig, ax = plt.subplots(1, 1, **{'figsize': (A4_width / 2, A4_height / 2)})
sns.countplot(
    x=filtered_data['Label'],
    data=filtered_data,
    hue=filtered_data['Scan_direction'],
    linewidth=0.5,
    palette='deep',
    edgecolor='black',
    ax=ax)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], fontsize='xx-small')
ax.set_xticklabels(
    ax.get_xticklabels(), fontsize='xx-small', rotation=45, ha='right')
ax.set_xlabel('')
ax.set_ylabel('Number of working pixels')

# save figure and add to powerpoint
image_path = ntpath.join(image_folder, f'boxchart_yields.png')
fig.savefig(image_path)
data_slide.shapes.add_picture(
    image_path, left=lefts[str(0)], top=tops[str(0)], height=height)

# # Then it needs to be grouped by variable value. Each of these groupings is
# # appended to a list that is iterated upon later to generate the plots.
# # Create variables holding a dictionary for accessing lists of data for
# # boxplots.
# boxplotdata_HL = rgl.boxplotdata(grouped_by_var_HL)
# boxplotdata_LH = rgl.boxplotdata(grouped_by_var_LH)
#
# # Iterate through the lists of grouped data to produce boxplots. Each plot
# # will contain all data from all values of a variable including a 'Control'
# # sample if one is given. Multiple plots are created if there is more than
# # one variable.
# print('Plotting boxplots and barcharts...', end='', flush=True)
# for i in range(len(boxplotdata_HL['var_names'])):
#     if boxplotdata_HL['var_names'][i] != 'Control':
#         for j in range(len(boxplotdata_HL['names_var'])):
#             if boxplotdata_HL['names_var'][j][0] == 'Control':
#                 boxplotdata_HL['names_var'][i].append(
#                     boxplotdata_HL['names_var'][j][0])
#                 boxplotdata_HL['Jsc_var'][i].append(
#                     boxplotdata_HL['Jsc_var'][j][0])
#                 boxplotdata_HL['Voc_var'][i].append(
#                     boxplotdata_HL['Voc_var'][j][0])
#                 boxplotdata_HL['FF_var'][i].append(
#                     boxplotdata_HL['FF_var'][j][0])
#                 boxplotdata_HL['PCE_var'][i].append(
#                     boxplotdata_HL['PCE_var'][j][0])
#                 boxplotdata_HL['Rs_var'][i].append(
#                     boxplotdata_HL['Rs_var'][j][0])
#                 boxplotdata_HL['Rsh_var'][i].append(
#                     boxplotdata_HL['Rsh_var'][j][0])
#
#         # Build a list of x values for scatter plots that will overlay
#         # boxplots.
#         x = []
#         for k in range(1, 1 + len(boxplotdata_HL['names_var'][i])):
#             x.extend([k] * len(boxplotdata_HL['Jsc_var'][i][k - 1]))
#
#         # Create new slide and box plots for PV parameters, save figures,
#         # and add them to the new slide
#         data_slide = rgl.title_image_slide(
#             prs, boxplotdata_HL['var_names'][i] + ' basic parameters')
#         params = ['Jsc', 'Voc', 'FF', 'PCE']
#         for ix, p in enumerate(params):
#             image_path = image_folder + 'boxplot_' + p + '.png')
#             rgl.create_save_boxplot(p, boxplotdata_HL, boxplotdata_LH, i, x,
#                                     image_path)
#             data_slide.shapes.add_picture(
#                 image_path,
#                 left=lefts[str(ix)],
#                 top=tops[str(ix)],
#                 height=height)
#
#         # Create new slide, box plots, and bar charts for parameters, save
#         # figures and add them to the new slide
#         data_slide = rgl.title_image_slide(
#             prs, boxplotdata_HL['var_names'][i] +
#             ' series and shunt resistances; yields')
#         params = ['Rs', 'Rsh', 'yields_var', 'yields_var_lab']
#         for ix, p in enumerate(params):
#             if p.find('yield') == -1:
#                 image_path = image_folder + 'boxplot_' + p + '.png')
#                 rgl.create_save_boxplot(p, boxplotdata_HL, boxplotdata_LH, i,
#                                         x, image_path)
#             elif p == 'yields_var':
#                 image_path = image_folder + 'barchart_' + p + '.png')
#                 rgl.create_save_barchart(yields_var, names_yield_var, i,
#                                          image_path)
#             elif p == 'yields_var_lab':
#                 image_path = image_folder + 'barchart_' + p + '.png')
#                 rgl.create_save_barchart(yields_var_lab, names_yield_var_lab,
#                                          i, image_path)
#             data_slide.shapes.add_picture(
#                 image_path,
#                 left=lefts[str(ix)],
#                 top=tops[str(ix)],
#                 height=height)
#
# # Clear all figures from memory
# plt.close('all')
print('Done')

# Group data by label and sort ready to plot graph of all pixels per substrate
print('Plotting JV curves...', end='', flush=True)
re_sort_data = filtered_data.sort_values(
    ['Label', 'Pixel'], ascending=[True, True])
grouped_by_label = re_sort_data.groupby('Label')

# Define a colormap for JV plots
cmap = plt.cm.get_cmap('viridis')

# Create lists of varibales, values, and labels for labelling figures
substrates = re_sort_data.drop_duplicates(['Label'])
variables = list(substrates['Variable'])
values = list(substrates['Value'])
labels = list(substrates['Label'])

# Create figures, save images and add them to powerpoint slide
i = 0
for name, group in grouped_by_label:
    # Create a new slide after every four graphs are produced
    if i % 4 == 0:
        data_slide = rgl.title_image_slide(
            prs, f'JV scans of every working pixel, page {int(i / 4)}')

    # Create figure, axes, y=0 line, and title
    fig = plt.figure(figsize=(A4_width / 2, A4_height / 2), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(0, lw=0.5, c='black')
    ax.axvline(0, lw=0.5, c='black')
    ax.set_title(
        f'{labels[i]}, {variables[i]}, {values[i]}',
        fontdict={'fontsize': 'xx-small'})

    # get parameters for plot formatting
    c_div = 1 / (len(group) / 2)
    pixels = list(group['Pixel'])
    max_group_jsc = np.max(np.absolute(group['Jsc_int']))
    max_group_voc = np.max(np.absolute(group['Voc_int']))

    signs, counts = np.unique(np.sign(group['Jsc']), return_counts=True)
    if len(signs) == 1:
        sign = signs[0]
    else:
        ix = np.argmax(counts)
        sign = signs[ix]

    # Import data for each pixel and plot on axes
    j = 0
    for file, scan_dir in zip(group['Rel_Path'], group['Scan_direction']):
        if scan_dir == 'LH':
            data_LH = np.genfromtxt(file, delimiter='\t')
            data_LH = data_LH[~np.isnan(data_LH).any(axis=1)]
            ax.plot(
                data_LH[:, 0],
                data_LH[:, 1],
                label=pixels[j],
                c=cmap(pixels[j] * c_div),
                lw=2.0)
        elif scan_dir == 'HL':
            data_HL = np.genfromtxt(file, delimiter='\t')
            data_HL = data_HL[~np.isnan(data_HL).any(axis=1)]
            ax.plot(data_HL[:, 0], data_HL[:, 1], c=cmap(pixels[j] * c_div), lw=2.0)

        j += 1

    # Format the axes
    ax.set_xlabel('Applied bias (V)')
    ax.set_ylabel('J (mA/cm^2)')
    if sign > 0:
        ax.set_xlim([np.min(data_HL[:, 0]), max_group_voc + 0.25])
        ax.set_ylim([-max_group_jsc * 2.2, max_group_jsc * 1.1])
    else:
        ax.set_xlim([-max_group_voc - 0.25, np.max(data_HL[:, 0])])
        ax.set_ylim([-max_group_jsc * 1.1, max_group_jsc * 2.2])

    # Adjust plot width to add legend outside plot area
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    handles, labs = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labs, loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')

    # Format the figure layout, save to file, and add to ppt
    image_path = ntpath.join(image_folder, f'jv_all_{labels[i]}.png')
    fig.savefig(image_path, bbox_extra_artists=(lgd, ), bbox_inches='tight')
    data_slide.shapes.add_picture(
        image_path,
        left=lefts[str(i % 4)],
        top=tops[str(i % 4)],
        height=height)

    # Close figure
    plt.close(fig)

    i += 1

# filter dataframe to leave only the best pixel for each variable value
sort_best_pixels = filtered_data.sort_values(
    ['Variable', 'Value', 'PCE'], ascending=[True, True, False])
best_pixels = sort_best_pixels.drop_duplicates(['Variable', 'Value'])

# get parameters for defining position of figures in subplot, attempting to
# make it as square as possible
no_of_subplots = len(best_pixels['File_Path'])
subplot_rows = np.ceil(no_of_subplots**0.5)
subplot_cols = np.ceil(no_of_subplots / subplot_rows)

# create lists of varibales and values for labelling figures
variables = list(best_pixels['Variable'])
values = list(best_pixels['Value'])
jscs = list(np.absolute(best_pixels['Jsc']))

# Loop for iterating through best pixels dataframe and picking out JV data
# files. Each plot contains forward and reverse sweeps, both light and dark.
i = 0
for file, scan_dir in zip(best_pixels['Rel_Path'],
                          best_pixels['Scan_direction']):
    # Create a new slide after every four graphs are produced
    if i % 4 == 0:
        data_slide = rgl.title_image_slide(
            prs, f'Best pixel JVs, page {int(i / 4)}')

    # Create figure, axes, y=0 line, and title
    fig = plt.figure(figsize=(A4_width / 2, A4_height / 2), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(0, lw=0.5, c='black')
    ax.axvline(0, lw=0.5, c='black')
    ax.set_title(
        f'{labels[i]}, {variables[i]}, {values[i]}',
        fontdict={'fontsize': 'xx-small'})

    # Import data for each pixel and plot on axes, ignoring errors. If
    # data in a file can't be plotted just ignore it.
    if scan_dir == 'LH':
        JV_light_LH_path = file
        if file.endswith('liv1'):
            JV_light_HL_path = file.replace('liv1', 'liv2')
        elif file.endswith('liv2'):
            JV_light_HL_path = file.replace('liv2', 'liv1')
    elif scan_dir == 'HL':
        JV_light_HL_path = file
        if file.endswith('liv1'):
            JV_light_LH_path = file.replace('liv1', 'liv2')
        elif file.endswith('liv2'):
            JV_light_LH_path = file.replace('liv2', 'liv1')

    try:
        JV_light_LH_data = np.genfromtxt(JV_light_LH_path, delimiter='\t')
        JV_light_HL_data = np.genfromtxt(JV_light_HL_path, delimiter='\t')
        JV_dark_LH_data = np.genfromtxt(
            JV_light_LH_path.replace('liv', 'div'), delimiter='\t')
        JV_dark_HL_data = np.genfromtxt(
            JV_light_HL_path.replace('liv', 'div'), delimiter='\t')
    except OSError:
        pass

    JV_light_LH_data = JV_light_LH_data[
        ~np.isnan(JV_light_LH_data).any(axis=1)]
    JV_light_HL_data = JV_light_HL_data[
        ~np.isnan(JV_light_HL_data).any(axis=1)]

    try:
        JV_dark_LH_data = JV_dark_LH_data[
            ~np.isnan(JV_dark_LH_data).any(axis=1)]
        JV_dark_HL_data = JV_dark_HL_data[
            ~np.isnan(JV_dark_HL_data).any(axis=1)]
    except NameError:
        pass

    ax.plot(
        JV_light_LH_data[:, 0],
        JV_light_LH_data[:, 1],
        label='L->H',
        c='red',
        lw=2.0)
    ax.plot(
        JV_light_HL_data[:, 0],
        JV_light_HL_data[:, 1],
        label='H->L',
        c='black',
        lw=2.0)
    try:
        ax.plot(
            JV_dark_LH_data[:, 0],
            JV_dark_LH_data[:, 1],
            label='L->H',
            c='orange',
            lw=2.0)
        ax.plot(
            JV_dark_HL_data[:, 0],
            JV_dark_HL_data[:, 1],
            label='H->L',
            c='blue',
            lw=2.0)
    except NameError:
        pass

    # Format the axes
    ax.set_xlabel('Applied bias (V)')
    ax.set_ylabel('J (mA/cm^2)')
    ax.set_xlim(
        [np.min(JV_light_HL_data[:, 0]),
         np.max(JV_light_HL_data[:, 0])])
    ax.set_ylim([-jscs[i] * 1.1, jscs[i] * 1.1])
    ax.legend(loc='best')

    # Format the figure layout, save to file, and add to ppt
    image_path = ntpath.join(image_folder, f'jv_best_{variables[i]}_{variables[i]}.png')
    fig.tight_layout()
    fig.savefig(image_path)
    data_slide.shapes.add_picture(
        image_path,
        left=lefts[str(i % 4)],
        top=tops[str(i % 4)],
        height=height)

    # Close figure
    plt.close(fig)

    i += 1

# Sort and filter data ready for plotting different scan rates/repeat scans
sorted_data_scan = data.sort_values(
    ['Variable', 'Value', 'Label', 'Pixel', 'scan_num'],
    ascending=[True, True, True, True, True])
filtered_scan_HL = sorted_data_scan[(sorted_data_scan.Condition == 'Light')
                                    & (sorted_data_scan.FF_int > 0.1) &
                                    (sorted_data_scan.FF_int < 0.9) &
                                    (sorted_data_scan.Jsc_int > 0.01) &
                                    (sorted_data_scan.Scan_direction == 'HL')]
filtered_scan_LH = sorted_data_scan[(sorted_data_scan.Condition == 'Light')
                                    & (sorted_data_scan.FF_int > 0.1) &
                                    (sorted_data_scan.FF_int < 0.9) &
                                    (sorted_data_scan.Jsc_int > 0.01) &
                                    (sorted_data_scan.Scan_direction == 'LH')]

# Drop pixels only working in one scan direction
filtered_data_HL = filtered_data_HL[
    filtered_data_HL.Label.isin(filtered_data_LH.Label.values)
    & filtered_data_HL.Pixel.isin(filtered_data_LH.Pixel.values)]
filtered_data_LH = filtered_data_LH[
    filtered_data_LH.Label.isin(filtered_data_HL.Label.values)
    & filtered_data_LH.Pixel.isin(filtered_data_HL.Pixel.values)]

# Create groups of data for each pixel for a given label
group_by_label_pixel_HL = filtered_scan_HL.groupby(['Label', 'Pixel'])
group_by_label_pixel_LH = filtered_scan_LH.groupby(['Label', 'Pixel'])

# Iterate through these groups and plot JV curves if more than one scan
# has been performed
i = 0
for iHL, iLH in zip(group_by_label_pixel_HL.indices,
                    group_by_label_pixel_LH.indices):
    group_HL = group_by_label_pixel_HL.get_group(iHL)
    group_LH = group_by_label_pixel_LH.get_group(iLH)

    if any(int(scan) > 0 for scan in group_HL['scan_num']):
        # Get label, variable, value, and pixel for title and image path
        label = group_HL['Label'].unique()[0]
        variable = group_HL['Variable'].unique()[0]
        value = group_HL['Value'].unique()[0]
        pixel = group_HL['Pixel'].unique()[0]

        # Find maximum Jsc of the group for y-axis limits and number of
        # JV curves for division of the colormap for the curves
        jsc_max = max(max(group_HL['Jsc']), max(group_LH['Jsc']))
        c_div = 1 / len(group_HL)

        # Start a new slide after every 4th figure
        if i % 4 == 0:
            data_slide = rgl.title_image_slide(
                prs, f'Repeat scan JV curves, page {int(i / 4)}')

        # Create figure, axes, y=0 line, and title
        fig = plt.figure(figsize=(A4_width / 2, A4_height / 2), dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        ax.axhline(0, lw=0.5, c='black')
        ax.set_title(
            f'{labels[i]}, {variables[i]}, {values[i]}',
            fontdict={'fontsize': 'xx-small'})

        # Open data files and plot a JV curve on the same axes for each scan
        j = 0
        for path_HL, path_LH, scan_rate_HL, scan_rate_LH, scan_num_HL, scan_num_LH in zip(
                group_HL['Rel_Path'], group_LH['Rel_Path'],
                group_HL['Scan_rate'], group_LH['Scan_rate'],
                group_HL['scan_num'], group_LH['scan_num']):

            data_HL = np.genfromtxt(path_HL, delimiter='\t')
            data_LH = np.genfromtxt(path_LH, delimiter='\t')
            data_HL = data_HL[~np.isnan(data_HL).any(axis=1)]
            data_LH = data_LH[~np.isnan(data_LH).any(axis=1)]

            ax.plot(
                data_HL[:, 0],
                data_HL[:, 1],
                c=cmap(j * c_div),
                label=f'{scan_num_HL}, {scan_rate_HL} V/s')
            ax.plot(data_LH[:, 0], data_LH[:, 1], c=cmap(j * c_div))

            j += 1

        # Format the axes
        ax.set_xlabel('Applied bias (V)')
        ax.set_ylabel('J (mA/cm^2)')
        ax.set_xlim([np.min(data_HL[:, 0]), np.max(data_HL[:, 0])])
        ax.set_ylim([-jscs[i - 1] * 1.1, jscs[i - 1] * 1.1])

        # Adjust plot width to add legend outside plot area
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(
            handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

        # Format the figure layout, save to file, and add to ppt
        image_path = ntpath.join(image_folder, f'jv_repeats_{label}_{variable}_{value}_{pixel}.png')
        fig.savefig(image_path, bbox_extra_artists=(lgd), bbox_inches='tight')
        data_slide.shapes.add_picture(
            image_path,
            left=lefts[str(i % 4)],
            top=tops[str(i % 4)],
            height=height)

        # Close figure
        plt.close(fig)

        i += 1
print('Done')

# Build a max power stabilisation log dataframe from file paths and J-V log
# file.
print('Plotting stabilisation data...', end='', flush=True)
i = 0
for index, row in spo_data.iterrows():
    # Get label, variable, value, and pixel for title and image path
    label = row['Label']
    variable = row['Variable']
    value = row['Value']
    pixel = row['Pixel']
    vmp = row['Vmp_int']

    # Start a new slide after every 4th figure
    if i % 4 == 0:
        data_slide = rgl.title_image_slide(
            prs, f'Stabilised power output, page {int(i / 4)}')

    # Open the data file
    path = row['Rel_Path']
    spo = np.genfromtxt(path, delimiter='\t', skip_header=1, skip_footer=3)
    spo = spo[~np.isnan(spo).any(axis=1)]

    # Create figure object
    fig = plt.figure(figsize=(A4_width / 2, A4_height / 2), dpi=300)

    # Add axes for J and format them
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title(
        f'{label}, pixel {pixel}, {variable}, {value}, Vmp = {vmp} V',
        fontdict={'fontsize': 'xx-small'})
    ax1.scatter(
        spo[:, 0], np.absolute(spo[:, 1]), color='black', s=5, label='J')
    ax1.scatter(
        spo[:, 0],
        np.absolute(spo[:, 2]),
        color='red',
        s=5,
        marker='s',
        label='PCE')
    ax1.set_ylabel('|J| (mA/cm^2) or PCE (%)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylim(0)
    ax1.legend()

    # Format the figure layout, save to file, and add to ppt
    image_path = ntpath.join(image_folder, f'spo_{label}_{variable}_{value}_{pixel}.png')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    fig.savefig(image_path)
    data_slide.shapes.add_picture(
        image_path,
        left=lefts[str(i % 4)],
        top=tops[str(i % 4)],
        height=height)

    # Close figure
    plt.close(fig)

    i += 1
print('Done')

try:
    # Sort, filter, and group intensity dependent data
    print('Plotting intensity dependence...', end='', flush=True)
    sorted_data_int = data.sort_values(
        ['Label', 'Pixel', 'Intensity', 'PCE'],
        ascending=[True, True, True, False])
    filtered_data_int_HL = sorted_data_int[(
        sorted_data_int.Scan_direction == 'HL')]
    filtered_data_int_LH = sorted_data_int[(
        sorted_data_int.Scan_direction == 'LH')]
    filtered_data_int_HL = filtered_data_int_HL.drop_duplicates(
        ['Label', 'Pixel', 'Intensity'])
    filtered_data_int_LH = filtered_data_int_LH.drop_duplicates(
        ['Label', 'Pixel', 'Intensity'])
    group_by_label_pixel_HL = filtered_data_int_HL.groupby(['Label', 'Pixel'])
    group_by_label_pixel_LH = filtered_data_int_LH.groupby(['Label', 'Pixel'])

    # filter groups with only 1 intensity and re-group
    group_by_label_pixel_HL_filt = group_by_label_pixel_HL.filter(
        lambda x: len(x) > 1)
    group_by_label_pixel_HL_filt = group_by_label_pixel_LH.filter(
        lambda x: len(x) > 1)
    group_by_label_pixel_HL = group_by_label_pixel_HL_filt.groupby(
        ['Label', 'Pixel'])
    group_by_label_pixel_LH = group_by_label_pixel_HL_filt.groupby(
        ['Label', 'Pixel'])

    if (len(group_by_label_pixel_HL.groups) != 0) & (len(
            group_by_label_pixel_HL.groups) != 0):
        # Plot inensity dependent graphs if experiment data exists
        for ng_HL, ng_LH in zip(group_by_label_pixel_HL,
                                group_by_label_pixel_LH):
            # Unpack group name and data
            name_HL = ng_HL[0]
            group_HL = ng_HL[1]
            name_LH = ng_LH[0]
            group_LH = ng_LH[1]

            # Get label, variable, value, and pixel for title and image path
            label = group_HL['Label'].unique()[0]
            variable = group_HL['Variable'].unique()[0]
            value = group_HL['Value'].unique()[0]
            pixel = group_HL['Pixel'].unique()[0]

            # Perfom linear fit to intensity dependence of Jsc
            m_HL, c_HL, r_HL, p_HL, se_HL = sp.stats.linregress(
                group_HL['Intensity'] * 100, group_HL['Jsc_int'])
            m_LH, c_LH, r_LH, p_LH, se_LH = sp.stats.linregress(
                group_LH['Intensity'] * 100, group_LH['Jsc_int'])
            r_sq_HL = r_HL**2
            r_sq_LH = r_LH**2

            # Create new slide (do this every iteration of the loop because each
            # loop creates four graphs)
            data_slide = rgl.title_image_slide(
                prs,
                f'Intensity dependence {label}, {variable}, {value}, pixel {pixel}'
            )

            # Create intensity dependence of Jsc figure
            fig = plt.figure(figsize=(A4_width / 2, A4_height / 2), dpi=300)
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(
                group_HL['Intensity'] * 100,
                group_HL['Jsc_int'],
                c='blue',
                label=
                f'H->L, m={rgl.round_sig_fig(m_HL, 3)}, c={rgl.round_sig_fig(c_HL, 3)}, R^2={rgl.round_sig_fig(r_sq_HL, 3)}'
            )
            ax.scatter(
                group_LH['Intensity'] * 100,
                group_LH['Jsc_int'],
                c='red',
                label=
                f'L->H, m={rgl.round_sig_fig(m_LH, 3)}, c={rgl.round_sig_fig(c_LH, 3)}, R^2={rgl.round_sig_fig(r_sq_LH, 3)}'
            )

            # Adjust plot width to add legend outside plot area
            ax.legend(loc='upper left', scatterpoints=1, prop={'size': 9})

            # Plot linear fits
            ax.plot(
                group_HL['Intensity'] * 100,
                group_HL['Intensity'] * 100 * m_HL + c_HL,
                c='blue')
            ax.plot(
                group_LH['Intensity'] * 100,
                group_LH['Intensity'] * 100 * m_LH + c_LH,
                c='red')

            # Format axes
            ax.set_xlabel('Light intensity (mW/cm^2)')
            ax.set_ylabel('Jsc (mA/cm^2)')
            ax.set_xlim([0, np.max(group_HL['Intensity'] * 100) * 1.05])

            # Format the figure layout, save to file, and add to ppt
            image_path = ntpath.join(image_folder, f'intensity_jsc_{label}_{variable}_{value}_{pixel}.png')
            fig.tight_layout()
            fig.savefig(image_path)
            data_slide.shapes.add_picture(
                image_path,
                left=lefts[str(0)],
                top=tops[str(0)],
                height=height)

            # Close figure
            plt.close(fig)

            # Perfom linear fit to ln(Jsc) dependence of Voc to estimate n and
            # J0 assuming single diode equivalent circuit
            m_HL, c_HL, r_HL, p_HL, se_HL = sp.stats.linregress(
                np.log(group_HL['Jsc_int']), group_HL['Voc_int'])
            m_LH, c_LH, r_LH, p_LH, se_LH = sp.stats.linregress(
                np.log(group_LH['Jsc_int']), group_LH['Voc_int'])
            r_sq_HL = r_HL**2
            r_sq_LH = r_LH**2
            n_HL = m_HL * q / (kB * T)
            n_LH = m_LH * q / (kB * T)
            j0_HL = np.exp(-c_HL / m_HL)
            j0_LH = np.exp(-c_LH / m_LH)

            # Create ln(Jsc) dependence of Voc figure
            fig = plt.figure(figsize=(A4_width / 2, A4_height / 2), dpi=300)
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(
                np.log(group_HL['Jsc_int']),
                group_HL['Voc_int'],
                c='blue',
                label=
                f'H->L, n={rgl.round_sig_fig(n_HL, 3)}, J_0={j0_HL:.2e} (mA/cm^2), R^2={rgl.round_sig_fig(r_sq_HL, 3)}'
            )
            ax.scatter(
                np.log(group_LH['Jsc_int']),
                group_LH['Voc_int'],
                c='red',
                label=
                f'L->H, n={rgl.round_sig_fig(n_LH, 3)}, J_0={j0_LH:.2e} (mA/cm^2), R^2={rgl.round_sig_fig(r_sq_LH, 3)}'
            )

            # Adjust plot width to add legend outside plot area
            ax.legend(loc='upper left', scatterpoints=1, prop={'size': 9})

            # Plot linear fits
            ax.plot(
                np.log(group_HL['Jsc_int']),
                np.log(group_HL['Jsc_int']) * m_HL + c_HL,
                c='blue')
            ax.plot(
                np.log(group_LH['Jsc_int']),
                np.log(group_LH['Jsc_int']) * m_LH + c_LH,
                c='red')

            # Format axes
            ax.set_xlabel('ln(Jsc) (mA/cm^2)')
            ax.set_ylabel('Voc (V)')
            ax.set_ylim([
                np.min(group_HL['Voc_int']) * 0.95,
                np.max(group_HL['Voc_int']) * 1.05
            ])

            # Format the figure layout, save to file, and add to ppt
            image_path = ntpath.join(image_folder, f'intensity_voc_{label}_{variable}_{value}_{pixel}.png')
            fig.tight_layout()
            fig.savefig(image_path)
            data_slide.shapes.add_picture(
                image_path,
                left=lefts[str(1)],
                top=tops[str(1)],
                height=height)

            # Close figure
            plt.close(fig)

            # Create intensity dependence of FF figure
            fig = plt.figure(figsize=(A4_width / 2, A4_height / 2), dpi=300)
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(
                group_HL['Intensity'] * 100,
                group_HL['FF_int'],
                c='blue',
                label='H->L')
            ax.scatter(
                group_LH['Intensity'] * 100,
                group_LH['FF_int'],
                c='red',
                label='L->H')
            ax.legend(loc='best', scatterpoints=1)
            ax.set_xlabel('Light intensity (mW/cm^2)')
            ax.set_ylabel('FF')
            ax.set_xlim([0, np.max(group_HL['Intensity'] * 100) * 1.05])

            # Format the figure layout, save to file, and add to ppt
            image_path = ntpath.join(image_folder, f'intensity_ff_{label}_{variable}_{value}_{pixel}.png')
            fig.tight_layout()
            fig.savefig(image_path)
            data_slide.shapes.add_picture(
                image_path,
                left=lefts[str(2)],
                top=tops[str(2)],
                height=height)

            # Close figure
            plt.close(fig)

            # Create intensity dependence of PCE figure
            fig = plt.figure(figsize=(A4_width / 2, A4_height / 2), dpi=300)
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(
                group_HL['Intensity'] * 100,
                group_HL['PCE_int'],
                c='blue',
                label='H->L')
            ax.scatter(
                group_LH['Intensity'] * 100,
                group_LH['PCE_int'],
                c='red',
                label='L->H')
            ax.legend(loc='best', scatterpoints=1)
            ax.set_xlabel('Light intensity (mW/cm^2)')
            ax.set_ylabel('PCE (%)')
            ax.set_xlim([0, np.max(group_HL['Intensity'] * 100) * 1.05])

            # Format the figure layout, save to file, and add to ppt
            image_path = ntpath.join(image_folder, f'intensity_pce_{label}_{variable}_{value}_{pixel}.png')
            fig.tight_layout()
            fig.savefig(image_path)
            data_slide.shapes.add_picture(
                image_path,
                left=lefts[str(3)],
                top=tops[str(3)],
                height=height)

            # Close figure
            plt.close(fig)

        # Plot intensity dependent JV curves
        i = 0
        for ng_HL, ng_LH in zip(group_by_label_pixel_HL,
                                group_by_label_pixel_LH):
            # Unpack group name and data
            name_HL = ng_HL[0]
            group_HL = ng_HL[1]
            name_LH = ng_LH[0]
            group_LH = ng_LH[1]

            # Get label, variable, value, and pixel for title and image path
            label = group_HL['Label'].unique()[0]
            variable = group_HL['Variable'].unique()[0]
            value = group_HL['Value'].unique()[0]
            pixel = group_HL['Pixel'].unique()[0]

            # Find maximum Jsc of the group for y-axis limits and number of
            # JV curves for division of the colormap for the curves
            jsc_max = max(max(group_HL['Jsc']), max(group_LH['Jsc']))
            c_div = 1 / len(group_HL)

            # Start a new slide after every 4th figure
            if i % 4 == 0:
                data_slide = rgl.title_image_slide(
                    prs, f'Intensity dependent JV curves, page {int(i / 4)}')

            # Create figure, axes, y=0 line, and title
            fig = plt.figure(figsize=(A4_width / 2, A4_height / 2), dpi=300)
            ax = fig.add_subplot(1, 1, 1)
            ax.axhline(0, lw=0.5, c='black')
            ax.set_title(
                f'{label}, pixel {pixel}, {variable}, {value}',
                fontdict={'fontsize': 'xx-small'})

            # Open data files and plot a JV curve on the same axes for each scan
            j = 0
            for path_HL, path_LH, intensity_HL, intensity_LH in zip(
                    group_HL['Rel_Path'], group_LH['Rel_Path'],
                    group_HL['Intensity'], group_LH['Intensity']):

                data_HL = np.genfromtxt(path_HL, delimiter='\t')
                data_LH = np.genfromtxt(path_LH, delimiter='\t')
                data_HL = data_HL[~np.isnan(data_HL).any(axis=1)]
                data_LH = data_LH[~np.isnan(data_LH).any(axis=1)]

                ax.plot(
                    data_HL[:, 0],
                    data_HL[:, 1],
                    c=cmap(j * c_div),
                    label=f'{round(intensity_HL * 100, 1)} mW/cm^2')
                ax.plot(data_LH[:, 0], data_LH[:, 1], c=cmap(j * c_div))

                j += 1

            # Format axes
            ax.set_xlabel('Applied voltage (V)')
            ax.set_ylabel('J (mA/cm^2)')
            ax.set_xlim([np.min(data_LH[:, 0]), np.max(data_LH[:, 0])])
            ax.set_ylim([-jsc_max * 1.05, jsc_max * 1.05])

            # Adjust plot width to add legend outside plot area
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(
                handles,
                labels,
                loc='upper left',
                bbox_to_anchor=(1, 1),
                prop={'size': 9})

            # Format the figure layout, save to file, and add to ppt
            image_path = ntpath.join(image_folder, f'jv_intensity_{label}_{variable}_{value}_{pixel}.png')
            fig.savefig(
                image_path, bbox_extra_artists=(lgd, ), bbox_inches='tight')
            data_slide.shapes.add_picture(
                image_path,
                left=lefts[str(i % 4)],
                top=tops[str(i % 4)],
                height=height)

            # Close figure
            plt.close(fig)

            i += 1
except KeyError:
    pass
print('Done')

# export log file with extra analysis
print('Saving log file...', end='', flush=True)
data.to_csv(log_filepath.replace('.log', '_extra.log'))
print('Done')

# Save powerpoint presentation
print('Saving powerpoint presentation...', end='', flush=True)
prs.save(log_filepath.replace('.log', '_summary.pptx'))
print('Done')
