# Ben's thesis install guide

## Set up python

The following commands are for getting the correct Python modules using a linux terminal

Development is done with Python 3.6.9, if you do not have Python 3, it will need to be installed before going further.

If Python 3 is installed, use the following commands to get the additional libraries used by this project:

```python
# install pip3 and get the std stuff
sudo apt install python3-pip
pip3 install numpy
pip3 install networkx --user

# scipy is a dependency of PyAstronomy
pip3 install scipy
pip3 install PyAstronomy

# gui lib
pip3 install PyQt5

# openGL api
pip3 install vtk

# -------- optional --------- #

# numba is used to speed up some of the link calculation.
# if it is not installed, the program will default to
# basic numpy+python. if numba is installed, some of the
# link calculation will be optimized to machine code (making it faster)
pip3 install numba --user

# I use tabs for indentation, tabwidth set to 2
# this is an optional linter that I use in VS Code
pip3 install flake8-tabs

```

# Optional, not required for running simulation


## Set up VS code

* I use tabs for indentation and spaces for alignment! Bite me!
* tabwidth = 2
* Column width is limited to 80 chars.
* Docstrings are made using sciPy style.
* Linter is flake8-tabs (like flake8 , but allows tabs for indentation)
  * Settings › Python › Linting: Flake8 Args += '--use-flake8-tabs'
* running 'code spell checker' extension
* running python files from "thesis-git/"


___
## Set up tex

First get the full texlive package
The whole package is not used (maybe 5%) but it is much
easier to grab the whole thing then install one module at a time
```
sudo apt install texlive-full
```
To use tex in VS Code, install the 'latex-workshop' extension. Now you can open a .tex file and hit the little view pdf button... to make a pdf.

___


## Util commands

Create .gif files from some images:
```
sudo apt install imagemagick-6.q16
convert -delay 10 -loop 0 *.png gt_3.gif
```

Compress gif so it can be embedded in presentation
```
sudo apt install gifsicle
gifsicle -i input.gif -03 --colors 32 --resize 256x256 -o out_optomized.gif
```

If the gifs using the above come out bad, I found that using gimp works well:
* open gimp > file > open as layers: (select all the images you want in gif)
* file > export > "somefilename.gif" > pick the settings you want


___

## How to set up CUDA for Linux Mint 19

First, make sure you have the latest nvidia driver (nvidia-driver-440.33).

***nvidia toolkit 10.2 only works with nvidia driver 440.33 on linux***

test that the nvidia driver works with nvidia-smi command, you should see something like:

```bash
dev@dev:~$ nvidia-smi
Sun Dec  1 15:52:16 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GT 630      On   | 00000000:01:00.0 N/A |                  N/A |
| N/A   23C    P8    N/A /  N/A |      1MiB /  2002MiB |     N/A      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0                    Not Supported                                       |
+-----------------------------------------------------------------------------+

```


Once you have the driver, make sure you have this package for g++/gcc:
```
sudo apt install build-essential dkms
```

Next install the latest nvidia developer toolkit (10.2) following the instructions on nvidia's site: <https://developer.nvidia.com/cuda-downloads>

Make sure that your nvidia driver is correct for the toolkit version (toolkit 10.2 needs nvidia-driver-440.33). After installing the toolkit, add the cuda bin to your path.
```bash
# add this to your .bashrc
# location of bin may need to be changed
PATH=$PATH:/usr/local/cuda-10.2/bin
```

Now, you should be able to open your shell and check to see that the cuda compiler (nvcc) works with the following:
```bash
dev@dev:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_19:24:38_PDT_2019
Cuda compilation tools, release 10.2, V10.2.89
dev@dev:~$
```

Now, go back and pip-install pycuda

if you want pycuda...


