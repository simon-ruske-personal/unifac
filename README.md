# CUDA & Python UNIFAC

The following repository includes a Python and CUDA implementation of the UNIFAC model (https://en.wikipedia.org/wiki/UNIFAC). The primary purpose of the Python implementation was to outline the model before developing the CUDA implmentation.  

## Getting started

### Python implementation

#### Executing
Pass in a group flag and group stoich file as arguments e.g.
```
> python unifac.py test_files\group_flag_1.txt test_files\group_stoich_1.txt
```

### CUDA version - In Progress

This is not quite finished. At present the code will output the combinitorial.
TODO: Finish writing the residual functionality

#### Prerequisites
At the moment the implementation has been tested using the following:
* CUDA version 10.1
* VS 2019 Community Edition
* GTX 1070 (You will need a CUDA capible GPU)

#### Compilation
Run the following from the Native Tools Command Prompt
```
> nvcc unifac_i.cu -o unifac -l cublas
```

#### Executing
The following parameters need to be set in the config.txt file which are passed to unifac_i.txt
* Q, R - filenames for surface area and volume contributions taken from the literature
* Data2, Data_main - filenames for the interaction arameters and subgroup numbers, taken from the literature
* x, v - filenames for x and v, the mole fraction (x) and the number of each group (v) for each component 
* group_flag_array - filename that lists the groups which are used in v
* maxGroupNum, molecules - dimensions of v

To produce files for x, v, and the group_flag_array from a group_flag and group_stoich file you can run, for example:
```
> python unifac_reader.py test_files\group_flag_1.txt test_files\group_stoich_1.txt
```
which will read the files and produce x, v, and the group_flag_array, as is done in the python version, but without executing the model.
 
After the source has been compiled run
```
> unifac.exe config.txt
```
