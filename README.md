# DeepGerm-Deep learning platform

Yu-Peng Zhu (朱玉鹏) <sup>1 </sup>, Peng Wang(汪鹏)<sup>1*,*4 </sup>

<sup>1</sup>Nanjing Agricultural
University, State Key Laboratory of Crop Genetics and Germplasm Enhancement,
College of Resources and Environmental Sciences, Nanjing, Jiangsu 210095, China*


<sup>4</sup>Center for Agriculture
and Health, Academy for Advanced Interdisciplinary Studies, Nanjing
Agricultural University, Nanjing 210095, China*

<sup> * </sup>Correspondence for the source code:Yu-Peng Zhu([2023203050@stu.njau.edu.cn](mailto:2018101176@njau.edu.cn)) and Peng Wang ([p.wang3@naju.edu.cn](mailto:p.wang3@naju.edu.cn)) 

## Description of main documents

(1) Test1_data/Test2_data - Deep Learning Test Dataset.

(2) nets - Deep Learning Network Architecture

(3) Main_program - Main program, dynamic trait extraction

## Install Python, Anaconda and Libraries

Setting up the Python environment to run the image processing (*u*-XRF) source code

1. **Install Python**
   Getting Started: If you are new to Python, we recommend reading the official Python beginner's guide ([https://wiki.python.org/moin/BeginnersGuide](https://wiki.python.org/moin/BeginnersGuide)).
   Windows users: Download and install the latest stable version of Python 3 from the official Python website([https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)).
   Mac OS users: Download and install the latest version of Python 3 for Mac from the official Python website([https://www.python.org/downloads/mac-osx/](https://www.python.org/downloads/mac-osx/)).Note: macOS usually comes with Python pre-installed, but it is recommended to install the latest version to ensure compatibility.
2. **Installing the Anaconda Python Distribution**
   Installation guide: Visit the official Anaconda installation guide ([https://docs.continuum.io/anaconda/install](https://docs.continuum.io/anaconda/install)) for detailed installation instructions.
   Windows users: Follow the Windows Installation Guide ([https://docs.continuum.io/anaconda/install/windows](https://docs.continuum.io/anaconda/install/windows)).
   Mac OS users: Follow the macOS installation guide ([https://docs.continuum.io/anaconda/install/mac-os.html](https://docs.continuum.io/anaconda/install/mac-os.html)).
   Graphical Installer: You can download the graphical installer for Anaconda from the Anaconda download page([https://www.continuum.io/downloads](https://www.continuum.io/downloads)).
   Recommended: The latest Anaconda Python distribution is recommended, as it contains many of the libraries and tools needed for scientific computing and data analysis.
3. **Install the necessary packages**
   The image analysis pipeline may require the installation of some third-party libraries. With Anaconda, you can install these libraries via the conda command. Here are some recommended libraries and their version numbers (version numbers may need to be adjusted to suit):

    conda install torch = 1.13.1   opencv-python = 4.7.0.72  scikit-learn=1.2.1 scikit-image=0.19.3 matplotlib=3.7.0 pandas=1.5.3 numpy=1.23.5 scipy=1.10.0  seaborn=0.12.2  

Note: If some libraries are not available in conda's default channels or have version mismatches, you can try installing them using the pip command, but make sure that your conda environment is activated and that you know how to manage dependencies between conda and pip.