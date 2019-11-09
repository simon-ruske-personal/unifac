import os
import numpy as np
from unifac import UNIFAC, load_files

for i in range(4):
    group_file = os.path.join("test_files", "group_flag_{}.txt".format(i+1))
    stoich_file = os.path.join("test_files", "group_stoich_{}.txt".format(i+1))
    molecules, x, v, UFC_Data_Q, UFC_Data_R, UFC_Data_main, UFC_Data2, group_flag_array, maxGroupNum_int, T = load_files(group_file, stoich_file)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        gamma = UNIFAC(molecules, x, v, UFC_Data_Q, UFC_Data_R, UFC_Data_main, UFC_Data2, group_flag_array, maxGroupNum_int, T, i+1)
    print(gamma)
    