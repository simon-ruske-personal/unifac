import os, sys, time
import numpy as np
from scipy.sparse import coo_matrix

def correct_number_of_paremeters(argv, expected_number = 3):
  if len(argv) != expected_number:
    raise ValueError("Two files should be specified including a "
                     "flag file and stoich file.")
 
def raise_input_errors(errors):
  if len(errors) > 0:
    raise OSError(" ".join(errors))
 
def both_files_exist(file_name_flag, file_name_stoich):
  errors = []
  if not os.path.exists(file_name_flag):
    errors.append("The flag file specified (first argument) does not exist.")
    
  if not os.path.exists(file_name_stoich):
    errors.append("The stoich file specified (second argument) does not exist.")
    
  raise_input_errors(errors)
                     
def read_and_validate_input_parameters(argv):
  correct_number_of_paremeters(argv)
  file_name_flag, file_name_stoich = sys.argv[1], sys.argv[2]
  both_files_exist(file_name_flag, file_name_stoich)
  return file_name_flag, file_name_stoich

def initialise_input_file_lists():
  rows, cols, vals, group_flag_array = [], [], [], []
  return rows, cols, vals, group_flag_array

def get_number_of_rows_and_columns(flag_file, stoich_file):
  n_rows, n_cols = flag_file.readline().strip('\n').split()
  stoich_file.readline().strip('\n').split()
  n_rows, n_cols = int(n_rows), int(n_cols)
  return n_rows, n_cols

def split_line(line):
  return line[:-1].split(' ')

def load_data(flag_file, stoich_file):
  rows, cols, vals, group_flag_list = initialise_input_file_lists()
  for i, (flag_line, stoich_line) in enumerate(zip(flag_file, stoich_file)):
    for flag, stoich in zip(split_line(flag_line), split_line(stoich_line)):
      if flag != '00':
        rows.append(i)
        cols.append(int(flag))
        vals.append(int(stoich))
        if flag not in group_flag_list:
          group_flag_list.append(flag)
          
  group_flag_array = np.array(group_flag_list, 'int')
  group_flag_array.sort()
  return rows, cols, vals, group_flag_array
  
def read_input_files(file_name_flag, file_name_stoich):
  
  with open(file_name_flag, "r") as flag_file,\
       open(file_name_stoich, "r") as stoich_file:
    n_rows, n_cols = get_number_of_rows_and_columns(flag_file, stoich_file)
    rows, cols, vals, group_flag_array = load_data(flag_file, stoich_file)
       
  return n_rows, n_cols, rows, cols, vals, group_flag_array

def read_parameter_files():
	UFC_Data_Q = np.genfromtxt(os.path.join("model_parameters", "Q.txt"), dtype='float32')
	UFC_Data_R = np.genfromtxt(os.path.join("model_parameters", "R.txt"), dtype='float32')
	UFC_Data_main = np.genfromtxt(os.path.join("model_parameters", 'UFC_Data_main.txt'), dtype = 'int') 
	UFC_Data2 = np.genfromtxt(os.path.join("model_parameters", 'UFC_Data2.txt'), dtype = 'float32')
	return UFC_Data_Q, UFC_Data_R, UFC_Data_main, UFC_Data2

def construct_v(cols, rows, vals, group_flag_array, molecules, maxGroupNum_int):
	d = dict(zip(group_flag_array, range(len(group_flag_array))))
	for i in range(len(cols)):
		cols[i] = d[cols[i]]

	v = coo_matrix((vals, (rows, cols)), shape = (molecules, maxGroupNum_int))
	v = v.toarray()	
	return v

def load_files(file_name_flag, file_name_stoich):
	start = time.perf_counter()
	n_rows, n_cols, rows, cols, vals, group_flag_array = read_input_files(file_name_flag, file_name_stoich)
	UFC_Data_Q, UFC_Data_R, UFC_Data_main, UFC_Data2 = read_parameter_files()
	molecules = n_rows
	end = time.perf_counter()
	T=298.15
	x = np.ones(molecules) / molecules # 1/ molecules for each molecules
	maxGroupNum_int = len(group_flag_array)
	v = construct_v(cols, rows, vals, group_flag_array, molecules, maxGroupNum_int)
	print('Reading files and pre unifac: ', (end - start) * 1000, ' ms')
	return molecules, x, v, UFC_Data_Q, UFC_Data_R, UFC_Data_main, UFC_Data2, group_flag_array, maxGroupNum_int, T

def UNIFAC(molecules, x, v, UFC_Data_Q, UFC_Data_R, UFC_Data_main, \
           UFC_Data2, group_flag_array, maxGroupNum_int, T, validate = False):
	
	'''
	The combinitorial
	'''

	# Not all the UFC_Data is needed
	# so reduct Q and R to only the values that are required.
	
	Q = UFC_Data_Q[group_flag_array]
	R = UFC_Data_R[group_flag_array]
	
	# Find q_i and r_i
	q = np.dot(v, Q)
	r = np.dot(v, R)
	
	# Calculate denominators of theta and psi
	xdotr = np.dot(x, r)
	xdotq = np.dot(x, q)
	
	# It isn't necessary to calculate theta and psi
	# instead calculate theta/psi and psi/x
	# also calculate L
	
	psi_x = r / xdotr
	theta_psi = q / xdotq / psi_x
	
	z = 10 # z is often assumed to be 10
	L = z/2 * (r - q) - (r -1)
	xdotl = np.dot(x, L) # also needed for final term
	
	# calculate ln_gamma_c
	ln_gamma_c = np.log(psi_x) + z/2 * q * np.log(theta_psi) + L - psi_x * xdotl 
	

	if validate:
		if not os.path.exists(".python_validation_files"):
			os.mkdir(".python_validation_files")
		np.savetxt(os.path.join(".python_validation_files", "Q_{}.csv".format(validate)), Q, delimiter=',')
		np.savetxt(os.path.join(".python_validation_files", "R_{}.csv".format(validate)), R, delimiter=',')
		np.savetxt(os.path.join(".python_validation_files", "qdot_{}.csv".format(validate)), q, delimiter=',')
		np.savetxt(os.path.join(".python_validation_files", "rdot_{}.csv".format(validate)), r, delimiter=',')
		np.savetxt(os.path.join(".python_validation_files", "dots_{}.csv".format(validate)), [xdotr, xdotq, xdotl], delimiter=',')
		np.savetxt(os.path.join(".python_validation_files", "psi_x_{}.csv".format(validate)), psi_x, delimiter=',')
		np.savetxt(os.path.join(".python_validation_files", "theta_psi_{}.csv".format(validate)), theta_psi, delimiter=',')
		np.savetxt(os.path.join(".python_validation_files", "L_{}.csv".format(validate)), L, delimiter=',')
		np.savetxt(os.path.join(".python_validation_files", "ln_gamma_c_{}.csv".format(validate)), ln_gamma_c, delimiter=',')
	
	'''
	the residual 
	'''
	
	# find X and theta
	X = np.sum(v, 0) / np.sum(v)
	Theta = Q * X / sum(Q * X)
	
	
	X_i = v.transpose() / np.sum(v, 1)
	Theta_i = Q * X_i.transpose()  
	Theta_i = (Theta_i.transpose() / np.sum(Theta_i, 1)).transpose()
	
	# find a 
	needed = UFC_Data_main[group_flag_array]
	a = UFC_Data2[needed][:, needed]
	
	# find psi
	Psi = np.exp(-a/T)
	
	# find ln gamma _k_i
	thePsi = np.dot(Theta_i, Psi)
	thePsi_t = np.dot(Theta_i/thePsi, Psi.transpose())
	ln_gamma_k_i = Q * (1.0 - np.log(thePsi) - thePsi_t)
	
	# find ln gamma_k
	thePsi = np.dot(Theta, Psi)
	thePsi_t = np.dot(Theta/thePsi, Psi.transpose())
	ln_gamma_k = Q * (1.0 - np.log(thePsi) - thePsi_t)
	
	# residual
	ln_gamma_r = np.sum(v * (ln_gamma_k - ln_gamma_k_i), 1)
	
	# put it together
	ln_gamma = ln_gamma_r + ln_gamma_c
	
	gamma = np.exp(ln_gamma)
	return(gamma)

def main():
	file_name_flag, file_name_stoich = read_and_validate_input_parameters(sys.argv)
	molecules, x, v, UFC_Data_Q, UFC_Data_R, UFC_Data_main, UFC_Data2, group_flag_array, maxGroupNum_int, T = load_files(file_name_flag, file_name_stoich)
	start = time.perf_counter()
	with np.errstate(divide = 'ignore', invalid = 'ignore'):
		gamma = UNIFAC(molecules, x, v, UFC_Data_Q, UFC_Data_R, UFC_Data_main, UFC_Data2, group_flag_array, maxGroupNum_int, T)
	end = time.perf_counter()
	print('Running UNIFAC: ', (end-start) * 1000, 'ms')
	print(gamma)


if __name__ == "__main__":
	main()