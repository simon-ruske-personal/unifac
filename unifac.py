import os, sys
import numpy as np
from time import clock
from scipy.sparse import coo_matrix

def correct_number_of_paremeters():
  if len(sys.argv) != 3:
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
                     
def read_and_validate_input_parameters():
  correct_number_of_paremeters()
  file_name_flag, file_name_stoich = sys.argv[1], sys.argv[2]
  both_files_exist(file_name_flag, file_name_stoich)
  return file_name_flag, file_name_stoich

start = clock()
file_name_flag, file_name_stoich = read_and_validate_input_parameters()

rows = []
cols = []
vals = []
group_flag_array = []

with open(file_name_flag, "r") as f, open(file_name_stoich, "r") as g:
	n_rows, n_cols = f.readline().strip('\n').split()
	g.readline().strip('\n').split()
	n_rows, n_cols = int(n_rows), int(n_cols)
	for i, (x, y) in enumerate(zip(f, g)):
		for item_1, item_2 in zip(x[:-1].split(' '), y[:-1].split(' ')):
			if item_1 != '00':
				rows.append(i)
				cols.append(int(item_1))
				vals.append(int(item_2))
				if item_1 not in group_flag_array:
					group_flag_array.append(item_1)

group_flag_array = np.array(group_flag_array, 'int')
group_flag_array.sort()
d = dict(zip(group_flag_array, range(len(group_flag_array))))
for i in range(len(cols)):
	cols[i] = d[cols[i]]

molecules = n_rows
maxGroupNum_int = len(group_flag_array)	
T=298.15
v = coo_matrix((vals, (rows, cols)), shape = (molecules, maxGroupNum_int))
v = v.toarray()
			
x = np.ones(molecules) / molecules # 1/ molecules for each molecules
UFC_Data_Q = np.genfromtxt('Q.txt', dtype='float32')
UFC_Data_R = np.genfromtxt('R.txt', dtype='float32')
UFC_Data_main = np.genfromtxt('UFC_Data_main.txt', dtype = 'int') 
UFC_Data2 = np.genfromtxt('UFC_Data2.txt', dtype = 'float32')				
end = clock()	

print('Reading files and pre unifac: ', (end - start) * 1000, ' ms')

def UNIFAC(molecules, x, v, UFC_Data_Q, UFC_Data_R, UFC_Data_main, \
           UFC_Data2, group_flag_array, maxGroupNum_int, T):
	
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

start = clock()
with np.errstate(divide = 'ignore', invalid = 'ignore'):
    gamma = UNIFAC(molecules, x, v, UFC_Data_Q, UFC_Data_R, UFC_Data_main, UFC_Data2, group_flag_array, maxGroupNum_int, T)
end = clock()
print('Running UNIFAC: ', (end-start) * 1000, 'ms')
print(gamma)