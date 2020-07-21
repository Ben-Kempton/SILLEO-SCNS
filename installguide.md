# Install guide

## Set up python

The following commands are for getting the correct Python modules using a linux terminal

Development is done with Python 3.6.9, if you do not have Python 3, it will need to be installed before going further.

If Python 3 is installed, use the following commands to get the additional libraries used by this project:

```python
# install pip3 (if needed)
sudo apt install python3-pip

# you probably already have numpy...
pip3 install numpy

# lib for graph analysis
pip3 install networkx --user

# scipy is a dependency of PyAstronomy, for orbit calculation
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


```

# Optional, not required for running simulation


## Set up VS code

* I use tabs for indentation and spaces for alignment! Bite me!
* tabwidth = 2
* Column width is limited to 80 chars.
* Docstrings are made using sciPy style.
* Linter is flake8-tabs (like flake8 , but allows tabs for indentation)
  * Settings › Python › Linting: Flake8 Args += '--use-flake8-tabs'


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





