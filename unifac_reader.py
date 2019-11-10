
import sys
import numpy as np
from unifac import read_and_validate_input_parameters, load_files

def main():
    file_name_flag, file_name_stoich = read_and_validate_input_parameters(sys.argv)
    molecules, x, v, UFC_Data_Q, UFC_Data_R, UFC_Data_main, UFC_Data2, group_flag_array, maxGroupNum_int, T = load_files(file_name_flag, file_name_stoich)
    np.savetxt("x.txt", x, fmt="%f", delimiter=' ')
    np.savetxt("v.txt", v, fmt="%f", delimiter=' ')
    np.savetxt("group_flag_array.txt", group_flag_array, fmt = "%i", delimiter=' ')

if __name__ == "__main__":
    main()