# eDNAshboardOpen
free dashboard for eDNA studies


### Try on binder
####Cloud access: slower and might break with large files (504 error) 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sanchestm/eDNAshboardOpen/HEAD)
1. **Click on lauch binder link**
2. **wait until it opens jupyter notebook**
3. **Click in the ednadashboardapp.ipynb file**
4. **run all cells by pressing Alt+Enter twice or pressing the run button twice**
5. **click on the link output generated**

### Installing instruction
####Local install: faster, can run large files

1. **Download Anaconda and RapID source code**
    1. [Dowload Ananconda](https://www.anaconda.com/products/individual) if not done before
    2. Dowload and unzip the RapID-cell-counter manually: click the green button writen `code` (at the top center of this page) and then click `download zip` in the dropdown options (or use git clone if experienced)
2. **Open terminal**
    1. **In Windows** open Ananconda Navigator desktop app then click on CMD.exe Prompt
![screenshot](https://github.com/sanchestm/RapID-cell-counter/blob/master/images/navigator.png)
    2. **In Linux** the terminal can be open directly via CRTL+ALT+T
    3. **In Mac:** open terminal by searching `terminal` in Spotlight (or Finder). Open the terminal by clicking the terminal app
3. In the terminal copy-paste and press enter for the following code
    1. `conda create env --name eDNAshboardOpen --file environment.yml`


### Run program

1. **Open terminal**
2. In the terminal, activate conda environment copy-paste and press enter for the following code
```
conda activate eDNAshboardOpen
```

3. Once we activated the conda environment (which contains all the necessary packages to run the code) we can locate the file (the directory where we downloaded and unzipped the package) and enter the directory to be able to run the program. As an example if we unzipped our file in the Downloads directory we can open this directory using the `cd` Command. In Linux and Mac, the dashes are `/` while in windows we use `\`
```
cd Downloads\eDNAshboardOpen-master
```

4. Start the software by typing the following code into the terminal and pressing enter
```
python app.py
```
5. Then click on the links that appear on your terminal or open a new tab in your browser and access http://127.0.0.1:8096
6. Close website by pressing ctrl-c in the terminal util python is terminated

### Rerunning the program

To rerun the program once we closed it, we only have to reopen the terminal. Activate the RapID environment. Use the `cd` to navigate to the directory of the mainQT5.py file and the execute it using `python mainQT5.py`. Or run the following lines if the RapID source code is in Downloads:
```
conda activate eDNAshboardOpen
cd Downloads\eDNAshboardOpen-master
python app.py
```
