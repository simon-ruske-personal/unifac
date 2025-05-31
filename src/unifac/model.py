import numpy as np

from src.unifac.model_parameters import UnifacModelParameters

def calculate_combinatorial(
    v,
	Q,
	R,
	x,
):
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
	return np.log(psi_x) + z/2 * q * np.log(theta_psi) + L - psi_x * xdotl 

def calculate_residual(
	v,
	Q,
	a,
	T
):
	# find X and theta
	X = np.sum(v, 0) / np.sum(v)
	Theta = Q * X / sum(Q * X)
	
	X_i = v.transpose() / np.sum(v, 1)
	Theta_i = Q * X_i.transpose()  
	Theta_i = (Theta_i.transpose() / np.sum(Theta_i, 1)).transpose()
	
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
	return np.sum(v * (ln_gamma_k - ln_gamma_k_i), 1)

def calculate_unifac_coefficients(
	x, 
	v, 
	group_flag_array,
	temperature: float,
	parameters: UnifacModelParameters
):
	# Not all the UFC_Data is needed
	# so reduct Q and R to only the values that are required.
	Q = parameters.Q[group_flag_array]
	R = parameters.R[group_flag_array]
	ln_gamma_c = calculate_combinatorial(v, Q, R, x)

    # find a 
	needed = parameters.Data_main[group_flag_array]
	a = parameters.Data_2[needed][:, needed]
	
	ln_gamma_r = calculate_residual(v, Q, a, temperature)
	
	# put it together
	ln_gamma = ln_gamma_r + ln_gamma_c
	
	gamma = np.exp(ln_gamma)
	return gamma
