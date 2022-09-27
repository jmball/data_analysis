# data_analysis
Generate a report from solar simulator measurement data.

## Installation and Usage
### Windows and MacOS (non-Python users)
Download and run the latest release for you opterating system from [here](https://github.com/jmball/data_analysis/releases).

### Windows and MacOS (Python users)
Create and activate a new Python (version >= 3.6) virtual environment e.g. using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), [venv](https://docs.python.org/3/library/venv.html) etc. Then clone this repository using [git](https://git-scm.com) and navigate to its newly created directory:
```
git clone https://github.com/jmball/data_analysis.git
cd data_analysis
```
Install the dependencies into the virtual environment using:
```
pip install -r requirements.txt
```
To run the program with a GUI on Windows (or Linux) call:
```
python data_analysis.py
```
or on a MacOS call:
```
pythonw data_analysis.py
```
To skip the GUI use:
```
python data_analysis.py --ignore-gooey "[folder]" [fix_ymin_0]
```
where `[folder]` is the absolute path to the folder containing data and `fix_min_0` is a flag indicating whether you want the y-axes of boxplots to start from zero (`yes`), or to autoscale (`no`).

### Linux
First, install the wxPython prerequisites listed [here](https://github.com/wxWidgets/Phoenix#prerequisites).

In addition, if your distribution's package manager doesn't include tkinter with your Python installation (e.g. Ubuntu), it must be installed separately (e.g. `sudo apt install python3.x-tk`, where x denotes your version of python3).

Then follow the instructions for 'Windows and MacOS (Python users)' above.

## Build instructions
To compile the program into a standalone binary file first follow the 'Installation and Usage' instructions for 'Windows and MacOS (Python users)' or 'Linux' above until you have installed the dependencies from the `requirements.txt` file. Then run:
```
pyinstaller data_analysis.spec
```
This will create two new folders in the current directory called `build` and `dist`. The binary file is in the `dist` folder and will be called `data_analysis.exe` on Windows, `data_analysis.app` on MacOSX, and just `data_analysis` on Linux.

## Notes
### Data filtering in boxplots
Some filtering of the data used is attempted to exclude measurements from boxplots where devices weren't working or not contacted properly. The criteria for exclusion of a device are when any of the following are true:
- the quasi-FF estimated from steady-state scans is outside the range 0-1 (a solar cell must be in this range)
- the J<sub>SC</sub> estimated from steady-state scans > 50 mA/cm<sup>2</sup> (the maximum for solar cell with bandgap > ~1.0 eV)
- the FF estimated from J-V characteristics is outside the range 0-1 on either scan direction
- the J<sub>SC</sub> from J-V characteristics is < 0.01 mA/cm<sup>2</sup> on either scan direction (occurs when a device behaves as a resistor or was not contacted properly)

A summary of how many devices were included in the boxplots is given in the Yield plots. From this, one can infer how many devices were excluded.

### Steady-state measurement values
The parameter value returned from a steady-state measurement (parameter output as a function of time) is the mean value of the final 10 measurement points. It's important to make sure these scans do reach steady-state to have confidence that the output values are not transient.