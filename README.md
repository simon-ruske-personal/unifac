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
At the moment, the implementation expects, v.txt and x.txt to be present in the current directory.
 
After the source has been compiled run
```
> unifac.exe config.txt
```

TODO
* Finish adding the lines of code to read filenames for v and x from the config file
* Need to add a Python script to produce x and v from the input files without having to execute the Python UNIFAC function, so they can be passed through into the CUDA version.
