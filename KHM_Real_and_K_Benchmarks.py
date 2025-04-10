#----------- PACKAGES Required! ---------------#
import numpy as np
import scipy as sp
import math
from numpy.linalg import eigh, norm, svd       ##### To compute the Singular Value Decomposition
from matplotlib import pyplot as plt           ##### Package required to plot
import matplotlib as mpl
from scipy import stats
mpl.rcParams.update({"text.usetex":True})
plt.rcParams['figure.dpi'] = 100
#-------------- END of PACKAGES ---------------#
###############################################################################################################################

############ Define the primitive lattice vectors #####################
a1 = np.array([np.sqrt(3)/2, -3./2]) 
a2 = np.array([np.sqrt(3), 0.])

############ Reciprocal lattice vectors ###############################
b1 = (-4 * np.pi / (3)) * np.array([0, 1])
b2 = (2 * np.pi / (3)) * np.array([np.sqrt(3), 1])

###############################################################################################################################
##################################################### Twisted Boundary Conditions #############################################
###############################################################################################################################
########### Function that returns the twisted boundary condition if given the input t, which is the amount of twist ###########
###############################################################################################################################

def twist_ybc(L1, L2, t):                             ######## Twist for y-bonds
    '''
    ------------------------------------------------------------------------------------------------------
    This function : twist_ybc(L1, L2, t) splits out the coordinates for the Twisted Boundary Conditions
    with a finite twist in the y-bonds with magnitude "t";
    ======================================================
    Twist Parametr : t : datatype (int)
    ======================================================
    t = 0  : => PBC (Periodic Boundary Condition)
    t = -1  : => APBC (Anti-Periodic Boundary Condition)
    t >= 1  : => TBC (Twisted Boundary Condition)
    ------------------------------------------------------------------------------------------------------
    '''
    Ti = np.zeros(L1, dtype = int)
    Tf = np.zeros(L1, dtype = int)
    tb = np.zeros((L1, 2), dtype = int)
    assert (t >= 0), "Antiperiodic Boundary Condition with a twist is currently not available; use t >= 0"
    for y in range(0, L1):
        Ti[y] = y * L2
    for y in range(1, L1 + 1):
        Tf[y-1] = L1 * L2 + (y * L2) - 1
    # Twisted and Periodic Boundary Condition
    for y in range(0, L1):
        tb[y, 0] = Ti[y]
        tb[y, 1] = Tf[(y + t) % L1]
    return tb

#################################################################################################################################
######################################################## Real Space Hamiltonian #################################################
#################################################################################################################################

def Ham_0(L1, L2, t, jx, jy, jz):
    m = L1 * L2                      ################################################## Total number of unit cells in the lattice
    N = m * 2                        ## Total number of sites, including the sublattice points (each unit cell includes 2 sitess)
    H = np.zeros((N, N))             ####################### Defining the Null Matrix Beforehand as most of the entries will be 0
    ty = twist_ybc(L1, L2, t)        ############################## Importing the boundary conditions with a twist in y-direction
    
######################## Defining the Matrix elements according to the gauge choices for different links ########################
    for i in range(0, m):            #### For xx-link
        H[i, i+m] = +jx
        H[i+m, i] = -jx
    
    for j in range(1, m+1):          #### For yy-link
        if (j % L2 != 0):
            H[j, j+m-1] = +jy
            H[j+m-1, j] = -jy
        
    for k in range(L2, m):           #### For zz-link
        H[k, k+m-L2] = +jz
        H[k+m-L2, k] = -jz
    
##################################  Applying the PBC (Periodic Boundary Conditions)  ############################################
    for b in range(0, L2):           ###### For Vertical bonds (zz-links) no twist in z-direction for the moment
        H[b, N-L2+b] = +jz
        H[N-L2+b, b] = -jz

    for y in range(L1):              ###### For yy-bonds
        H[ty[y][0], ty[y][1]] = +jy
        H[ty[y][1], ty[y][0]] = -jy

    return H   

#################################################################################################################################
################################################# Real Space Energy Eigenvalue ##################################################
#################################################################################################################################


def E_0(L1, L2, t, jx, jy, jz):
    N = L1 * L2 * 2 
    H0 = Ham_0(L1, L2, t, jx, jy, jz)        
    ##### Determine Nu
    Nu = H0.shape[0] // 2
    ##### Split the matrix into four N x N blocks
    F = H0[:Nu, :Nu]                                            # Top-left block
    M = H0[:Nu, Nu:]                                            # Top-right block
    MT = (-1) * H0[Nu:, :Nu]                                    # Bottom-left block
    D = (-1) * H0[Nu:, Nu:]                                     # Bottom-right block
    u, s, vT = sp.linalg.svd(M, full_matrices = False)          # SVD of M
    # e, p = sp.linalg.eigh(M)
    
    return s                                                    # return only the singular values

#################################################################################################################################
################################################## Momentum Space Energy Eigenvalue #############################################
#################################################################################################################################


def E_K0(L1, L2, t, Jx, Jy, Jz):                                ####################### function to compute the K-space energy eigenvalues
  '''
  To evaluate the following expression (Momentum Space Eigenvalues) :
  f(k) = J_x e^{i q.n_1} + J_y e^{i q.n_2} + J_a
  '''
    def K(L1, L2, t):                                           ############################################# function for momentum values
        k = np.zeros((L1 * L2, 2), dtype = complex)
        i = 0
        for n1 in range(L1):
            for n2 in range(L2):
                # k[i] = (b1 / L1) * n1 + (b2 / L2) * (n2)                       ###### Change!!! (No Twist code)
                k[i] = (n1 / L1) * b1 + (n2 / L2 - (n1 * t) / (L1 * L2)) * b2    ###### Change!!! (With Twist Code)
                i = i + 1
        return k
    Q = K(L1, L2, t)
    ######## Compute the components of the f-vector 
    f = 1 * ( Jx * np.exp(1j * np.dot(Q, a1)) + Jy * np.exp(1j * np.dot(Q, a2)) + Jz )
    return np.abs(f)


#################################################################################################################################
################################# Parameters list for the KHM (Kitaev Honeycomb Model) ##########################################
#################################################################################################################################

L1 = 14
L2 = 14
jx = 1
jy = 1
jz = 1
t = 0

# S0, e0 = E_0(L1, L2, t, jx, jy, jz)
S0 = E_0(L1, L2, t, jx, jy, jz)
Ek = E_K0(L1, L2, t, jx, jy, jz)


Error = np.sort(S0) - np.sort(Ek)
Ratio = np.sort(S0) / np.sort(Ek)

# print(type(t))

#################################################################################################################################
################################ PLOTS of Real and K Space: KHM (Kitaev Honeycomb Model) ########################################
#################################################################################################################################

fig, ax = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(r"Kitaev Honeycomb Model : $L_1=L_2=10$, $J_x=J_y=J_z=1$ \& Twist : $t=0$", fontsize=16)
# Plot 1
ax[0,0].plot(np.sort(S0), color = 'red', label = 'singular values')
ax[0,0].set_title('Energy Spectrum (Finite Size Real Space Hamiltonian)')
ax[0,0].set_xlabel(r'$Sites$')
ax[0,0].set_ylabel(r'Singular Values : $S$')

# Plot 2
ax[0,1].plot(np.sort(Ek), label = 'Ek2')
ax[0,1].set_title('Energy Spectrum (Finite Size Momentum Space Hamiltonian)')
ax[0,1].set_xlabel(r'$Sites$')
ax[0,1].set_ylabel(r'Eigenvalues : $E$')

# Plot 3
# # ax[1,0].plot(np.sort(e0), label = r'Real Space Hamiltonian Eigenvalues ($E$)')
# ax[1,0].plot(Error, label = r'$S-E$')
# ax[1,0].plot(1/Ratio, label = r'$S/E$')
# ax[1,0].legend()
# ax[1,0].set_title(r'Energy Eigenvalue Spectrum')
# ax[1,0].set_xlabel(r'$Sites$')
# ax[1,0].set_ylabel(r'Eigenvalue $ : \epsilon$')

# Plot 4
ax[1,1].plot(Error)
ax[1,1].set_ylabel(r'$S-E$',  fontsize = 9)
ax[1,1].set_xlabel('$Sites$', fontsize = 10)
ax[1,1].set_title(r'Error Plot')

plt.tight_layout()
# plt.savefig("Real vs K-space Eigenvalues Plot.pdf", bbox_inches="tight")     # If you want to save the plot (in PDF format)
plt.show()

########################## END of the BENCHMARKING #################################
