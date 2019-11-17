import os
import subprocess
import shutil
import numpy as np
from datetime import datetime
from unifac_reader import read_files
from unifac import UNIFAC, load_files

def check_cuda_version_exists():
    if not os.path.exists("unifac.exe"):
        raise RuntimeError("unifac.exe is missing")

def get_validation_files():
    test_file_directory = os.path.join(os.curdir, "test_files")
    filename_bases = ["group_flag_{}.txt", "group_stoich_{}.txt"]
    return [os.path.join(test_file_directory, filename_base.format(idx + 1))
            for filename_base in filename_bases
            for idx in range(4)]

def check_validation_files_exist(validation_files):
    missing_files = []
    for validation_file_name in validation_files:
        if not os.path.exists(validation_file_name):
            print(validation_file_name)
            missing_files.append(validation_file_name)
    if len(missing_files) > 0:
        raise RuntimeError("A number of test files were missing: " + ",".join(missing_files))

def setup_files(group_file_name, group_stoich_filename):
    print('reading')
    return read_files(group_file_name, group_stoich_filename)

def perform_pre_validation_checks(validation_files):
    check_cuda_version_exists()
    check_validation_files_exist(validation_files)
    print("Pre-validation checks completed.")

def create_config_file(filename, molecules, maxGroupNum):
    with open(filename, 'w') as f:
        f.write("Q : model_parameters/Q.txt\n")
        f.write("R : model_parameters/R.txt\n")
        f.write("Data2 : model_parameters/UFC_Data2.txt\n")
        f.write("Data_main : model_parameters/UFC_Data_main.txt\n")
        f.write("Group_Flag : group_flag_array.txt\n")
        f.write("v : v.txt\n")
        f.write("x : x.txt\n")
        f.write("maxGroupNum : {}\n".format(maxGroupNum))
        f.write("molecules : {}\n".format(molecules))

def execute_cuda_code(config_filename):
    previous = os.listdir(os.path.join(".cuda_validation_files"))
    subprocess.call(["unifac.exe", config_filename])
    new = os.listdir(os.path.join(".cuda_validation_files"))
    directory = list(set(new).difference(set(previous)))
    if len(directory) > 1:
        raise RuntimeError("More than one cuda output directory was produced")
    elif len(directory) == 0:
        raise RuntimeError("No cuda output directory was created")
    else:
        return directory[0]



def execute_python_code(group_file, stoich_file, idx):
    molecules, x, v, UFC_Data_Q, UFC_Data_R, UFC_Data_main, UFC_Data2, group_flag_array, maxGroupNum_int, T = load_files(group_file, stoich_file)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        gamma = UNIFAC(molecules, x, v, UFC_Data_Q, UFC_Data_R, UFC_Data_main, UFC_Data2, group_flag_array, maxGroupNum_int, T, idx+1)

def files_exist(filename_1, filename_2):
    if not os.path.exists(filename_1):
        raise ValueError('Missing file {}'.format(filename_1))
    if not os.path.exists(filename_2):
        raise ValueError('Missing file {}'.format(filename_2))

def check_nans(x1, x2):
    nan1 = np.isnan(x1)
    nan2 = np.isnan(x2)
    return np.all(nan1 == nan2)
        
        


def check_difference(filename_1, filename_2):
    files_exist(filename_1, filename_2)
    x1 = np.genfromtxt(filename_1, delimiter=',')
    x2 = np.genfromtxt(filename_2)
    log_difference(filename_1, filename_2, np.nanmax(x1-x2), check_nans(x1, x2))

def log_difference(filename_1, filename_2, difference, nanscorrect):
    with open(os.path.join('.output', 'temp.txt'), 'a') as f:
        f.write("Comparing {} with {} maximum difference was {}\n".format(
            os.path.basename(filename_1), 
            os.path.basename(filename_2),
            difference))
        if not nanscorrect:
            f.write('Nans did not match between two results\n')

def check_dots_difference(filename_1, filename_2):
    files_exist(filename_1, filename_2)
    x1 = np.genfromtxt(filename_1, delimiter=',')
    x2 = np.genfromtxt(filename_2, delimiter=':')
    log_difference(filename_1, filename_2, np.nanmax(x1-x2[:, -1]), check_nans(x1, x2[:, -1]))


def match_directories(directory_1, directory_2, i):
    check_difference(os.path.join(directory_1, 'rdot_{}.csv'.format(i)), os.path.join(directory_2, 'rdot.txt'))
    check_difference(os.path.join(directory_1, 'qdot_{}.csv'.format(i)), os.path.join(directory_2, 'qdot.txt'))
    check_dots_difference(os.path.join(directory_1, 'dots_{}.csv'.format(i)), os.path.join(directory_2, 'dots.txt'))
    check_difference(os.path.join(directory_1, 'psi_x_{}.csv'.format(i)), os.path.join(directory_2, 'psi_x.txt'))
    check_difference(os.path.join(directory_1, 'theta_psi_{}.csv'.format(i)), os.path.join(directory_2, 'the_psi.txt'))  
    check_difference(os.path.join(directory_1, 'L_{}.csv'.format(i)), os.path.join(directory_2, 'L.txt'))
    check_difference(os.path.join(directory_1, 'ln_gamma_c_{}.csv'.format(i)), os.path.join(directory_2, 'ln_gamma_c.txt'))

def cleanup_config(config_filename):
    os.remove(config_filename)

def cleanup_directories():
    for directory in ['.cuda_validation_files', '.python_validation_files', '.config', '.output']:
        if os.path.exists(directory):
            shutil.rmtree(directory)

def create_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def create_directories():
    for directory in ['.cuda_validation_files', '.python_validation_files', '.config', '.output']:
        create_directory(directory)

def main():

    validation_files = get_validation_files()
    perform_pre_validation_checks(validation_files)
    cleanup_directories()
    create_directories()

    for idx in range(4):
        config_filename = os.path.join(
            ".config", 
            str(datetime.now()).replace(':', '_').replace(' ', '').replace('.', '_') + '.txt'
        )
        group_flag_filename = os.path.join("test_files", "group_flag_{}.txt".format(idx + 1))
        group_stoich_filename = os.path.join("test_files", "group_stoich_{}.txt".format(idx + 1))
        molecules, maxGroupNum = setup_files(group_flag_filename, group_stoich_filename)
        create_config_file(config_filename, molecules, maxGroupNum)
        cuda_output_directory = execute_cuda_code(config_filename)
        execute_python_code(group_flag_filename, group_stoich_filename, idx)
        cleanup_config(config_filename)
        match_directories('.python_validation_files', os.path.join(".cuda_validation_files", cuda_output_directory), idx+1)

    shutil.copy(os.path.join('.output', 'temp.txt'), "validation_report.txt")
    cleanup_directories()

    
if __name__ == "__main__":
    main()
