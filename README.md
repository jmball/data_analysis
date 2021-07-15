# data_analysis
Generate a report from solar simulator measurement data.

## Usage
### Non-Python users
Download and run the latest release for you opterating system from [here](https://github.com/jmball/data_analysis/releases).

### Python users
Create and activate a new Python (version >= 3.6 recommended) virtual environment e.g. using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), [venv](https://docs.python.org/3/library/venv.html) etc. Then clone this repository using [git](https://git-scm.com) and navigate to its newly created directory:
```
git clone https://github.com/jmball/data_analysis.git
cd data_analysis
```
Install the dependencies into the virtual environment using:
```
pip install -r requirements.txt
```
To run the program with a GUI call:
```
python data_analysis.py
```
Or to skip the GUI use:
```
python data_analysis.py --ignore-gooey "[folder]" [fix_ymin_0]
```
where `[folder]` is the absolute path to the folder containing data and `fix_min_0` is a flag indicating whether you want the y-axes of boxplots to start from zero (`yes`), or to autoscale (`no`).

#### Linux Prerequisites
In order to install wxPython included in the requirements.txt file it's necessary to first install the prerequisites detailed [here](https://github.com/wxWidgets/Phoenix#prerequisites). In addition, if your distribution's package manager doesn't include tkinter with your Python installation (e.g. Ubuntu), it must be installed separately (e.g. `sudo apt install python3.x-tk`, where x denotes your version of python3).

## Build instructions
To compile the program into a standalone binary file first install the Linux prerequisites above if required, then follow the usage instructions for Python users above until you have installed dependencies from the `requirements.txt` file. Then run:
```
pyinstaller data_analysis.spec
```
This will create two new folders in the current directory called `build` and `dist`. The binary file is in the `dist` folder and will be called `data_analysis.exe` on Windows, `data_analysis.app` on MacOSX, and just `data_analysis` on Linux.