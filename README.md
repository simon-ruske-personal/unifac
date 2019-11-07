# CUDA & Python UNIFAC

The following repository includes a Python and CUDA implementation of the UNIFAC model (https://en.wikipedia.org/wiki/UNIFAC). The primary purpose of the Python implementation was to outline the model before developing the CUDA implmentation.  

## Getting started

### Python implementation

#### Prerequisites
* I prefer using Anaconda to run the Python code

#### Executing
Pass in a group flag and group stoich file as arguments e.g.
```
> python unifac.py test_files\group_flag_1.txt test_files\group_stoich_1.txt
```

### CUDA version

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
At the moment, the implementation expects, Q.txt, R.txt, UFC_Data2.txt, UFC_Data_main.txt, v.txt and x.txt to be present in the current directory.
 
After the source has been compiled run
```
> unifac.exe
```

TODO: allow for arguments to be passed in for the file names similar to functionality in the Python implmenetation
