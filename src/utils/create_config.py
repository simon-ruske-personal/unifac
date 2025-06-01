from pathlib import Path
import numpy as np
from utils.load_test_files import load_files


def create_config_file(filename, molecules, maxGroupNum):
    with open(filename, 'w') as f:
        f.write("Q : unifac/model_parameters/Q.txt\n")
        f.write("R : unifac/model_parameters/R.txt\n")
        f.write("Data2 : unifac/model_parameters/UFC_Data2.txt\n")
        f.write("Data_main : unifac/model_parameters/UFC_Data_main.txt\n")
        f.write("Group_Flag : group_flag.txt\n")
        f.write("v : v.txt\n")
        f.write("x : x.txt\n")
        f.write("maxGroupNum : {}\n".format(maxGroupNum))
        f.write("molecules : {}\n".format(molecules))

def create_x_and_v():
    test_directory = Path("tests")/ "test_files"
    group_file = test_directory / "group_flag_183_83.txt"
    group_stoich = test_directory / "group_stoich_183_83.txt"
    v, group_flag = load_files(group_file, group_stoich)
    np.savetxt("v.txt", v, delimiter = ' ')
    np.savetxt("group_flag.txt", group_flag.astype('int32'), delimiter=' ', fmt = '%i')
    np.savetxt("x.txt", np.ones(len(v)) / len(v), delimiter=' ')

if __name__ == "__main__":
    create_x_and_v()
    create_config_file("config.txt", 183, 83)