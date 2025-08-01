#----------- PACKAGES Required! ---------------#
import numpy as np
import scipy as sp
import math
from numpy.linalg import eigh, norm, svd                                    ##### To compute the Singular Value Decomposition
from matplotlib import pyplot as plt                                        ##### Package required to plot
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

def twist_ybc(L1, L2, t):                                                                            ######## Twist for y-bonds
    '''
    -------------------------------------------------------------------------------------------------------
    This function : ``twist_ybc(L1, L2, t)`` splits out the coordinates for the Twisted Boundary Conditions
    with a finite twist in the y-bonds with magnitude ``t`` for a Honeycomb Lattice of dimension L1 x L2;
    =======================================================================================================
    Twist Parametr : ``t`` : datatype (int)
    =======================================================================================================
    t = 0  : => PBC  (Periodic Boundary Condition)
    t = -1 : => APBC (Anti-Periodic Boundary Condition) [Currently not available]
    t >= 1 : => TBC  ((non-zero) Twisted Boundary Condition)
    -------------------------------------------------------------------------------------------------------
    '''
    Ti = np.zeros(L1, dtype = int)
    Tf = np.zeros(L1, dtype = int)
    tb = np.zeros((L1, 2), dtype = int)
    assert (t >= 0), "Antiperiodic Boundary Condition with a twist is currently not available; use t >= 0"
    for y in range(0, L1):
        Ti[y] = y * L2
    for y in range(1, L1 + 1):
        Tf[y-1] = L1 * L2 + (y * L2) - 1
    #### Twisted and Periodic Boundary Conditions ####
    for y in range(0, L1):
        tb[y, 0] = Ti[y]
        tb[y, 1] = Tf[(y + t) % L1]
    return tb

#################################################################################################################################
############################################## Real Space Hamiltonian for 0 flux state ##########################################
#################################################################################################################################

def Ham_0(L1, L2, t, jx, jy, jz):
    m = L1 * L2                      ################################################## Total number of unit cells in the lattice
    N = m * 2                        ### Total number of sites, including the sublattice points (each unit cell includes 2 sites)
    H = np.zeros((N, N))             ###################### Defining the Null Matrix Beforehand, as most of the entries will be 0
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
    for b in range(0, L2):           ###### For Vertical bonds (zz-links), no twist in z-direction for the moment
        H[b, N-L2+b] = +jz
        H[N-L2+b, b] = -jz

    for y in range(L1):              ###### For yy-bonds
        H[ty[y][0], ty[y][1]] = +jy
        H[ty[y][1], ty[y][0]] = -jy

    return H   

#################################################################################################################################
################################################# Real Space Singularvalues #####################################################
#################################################################################################################################


def E_0(L1, L2, t, jx, jy, jz):
    N = L1 * L2 * 2 
    H0 = Ham_0(L1, L2, t, jx, jy, jz)        
    ##### Determine Nu
    Nu = H0.shape[0] // 2
    ##### Split the matrix into four N x N blocks
    F = H0[:Nu, :Nu]                                            #### Top-left block
    M = H0[:Nu, Nu:]                                            #### Top-right block
    MT = (-1) * H0[Nu:, :Nu]                                    #### Bottom-left block
    D = (-1) * H0[Nu:, Nu:]                                     #### Bottom-right block
    u, s, vT = sp.linalg.svd(M, full_matrices = False)          #### SVD of M
    # e, p = sp.linalg.eigh(M)
    
    return s                                                    #### return only the singular values


##########################################################################################################################################
############################################## Real Space Hamiltonian for 2 flux state ###################################################
##########################################################################################################################################
# mu = 0                                      # Spin Flip at mu-th unit cell (z-type unit cell)!: mu = 0 implies (4,16) for 4 x 4 lattice! 
def Ham_2(L1, L2, t, jx, jy, jz, mu):
    m = L1 * L2                               # Total number of unit cells in the lattice
    N = m * 2                                 # Total number of sites, including the sublattice points (each unit cell includes 2 sites)
    H = np.zeros((N, N))                      # Defining the Null Matrix Beforehand, as most of the entries will be 0
    ty = twist_ybc(L1, L2, t)                 # Importing the boundary conditions with a twist in y-direction
    
######### Defining the Matrix elements according to the gauge choices for different links
    for i in range(0, m):                     # For xx-link
        H[i, i+m] = +jx
        H[i+m, i] = -jx
    
    for j in range(1, m+1):                   # For yy-link
        if (j % L2 != 0):
            H[j, j+m-1] = +jy
            H[j+m-1, j] = -jy
        
    for k in range(L2, m):                    # For zz-link
        H[k, k+m-L2] = +jz
        H[k+m-L2, k] = -jz
    
###### Applying the PBC (Periodic Boundary Conditions)
    for b in range(0, L2):                    # For Vertical bonds (zz-links), no twist in z-direction for the moment
        H[b, N-L2+b] = +jz
        H[N-L2+b, b] = -jz

    for y in range(L1):                       # For yy-bonds
        H[ty[y][0], ty[y][1]] = +jy
        H[ty[y][1], ty[y][0]] = -jy

############ Introducing the Single Spin Flip terms in z-bonds! (In the bulk, Boundary Conditions also included!)
    if (mu < m-L2):                           # For bulk z-spin-flip
        H[L2+mu, m+mu] = -jz
        H[m+mu, L2+mu] = +jz

    else:                                     # Boundary conditions for z-spin-flip
        H[L2+mu-m, m+mu] = -jz
        H[m+mu, L2+mu-m] = +jz
           
    return H   

##########################################################################################################################################
##################################################### Real Space Singularvalues ##########################################################
##########################################################################################################################################

def E_2f(L1, L2, t, jx, jy, jz, mu):
    N = L1 * L2 * 2 
    H2 = Ham_2(L1, L2, t, jx, jy, jz, mu)        
    #### Determine Nu
    Nu = H2.shape[0] // 2
    #### Split the matrix into four (N x N) blocks
    F = H2[:Nu, :Nu]                                               #### Top-left block
    M = H2[:Nu, Nu:]                                               #### Top-right block
    MT = (-1) * H2[Nu:, :Nu]                                       #### Bottom-left block
    D = (-1) * H2[Nu:, Nu:]                                        #### Bottom-right block
    u2, s2, v2T = sp.linalg.svd(M, full_matrices = False)          #### SVD of M
    
    return s2

##########################################################################################################################################
################# Ground State Energy per unit cell in the 0-flux state (Calculations for Periodic Boundary Conditions) ##################
##########################################################################################################################################
##################### E_gs = -\sum_{n=1}^m s_n : where s_n are singular values and E_gs is the ground state energy #######################

GE0p = []                                           ### Empty List to store the Ground State Energy per unit cell of a 0-flux state in PBC
L0p = 20                                            ### Maximum System Size
t0p = 0                                             ### Twist Parameter (= 0 implies PBC )
jx = jy = jz = 1                                    ### Model Parameters (Ising interaction Strengths)
sys_size0p = np.linspace(3, L0p, L0p-2).astype(int)
for l in sys_size0p:
    sing_0 = E_0(l, l, t0p, jx, jy, jz)             ### Extracting the singular values
    GE0p.append(-1 * np.sum(sing_0) / (1 * l*l))    ### Storing the ground state energy for different system sizes in the list
GE0p = np.array(GE0p)

##########################################################################################################################################
################# Relative Ground State Energy in the 0-flux state (Calculations for Periodic Boundary Conditions) #######################
##########################################################################################################################################

rE0p = []
for i, l in enumerate(sys_size0p):
    rE0p.append((GE0p[i]*l*l + l*l * 1.5745974))
rE0p = np.array(rE0p)

##########################################################################################################################################
################# Ground State Energy per unit cell in the 0-flux state (Calculations for Twisted Boundary Conditions) ###################
##########################################################################################################################################

GE0t = []                                           ### Empty List to store the Ground State Energy per unit cell of a 0-flux state in TBC
L0t = 20                                            ### Maximum System Size
t0t = 1                                             ### Twist Parameter (= 1 implies TBC )
jx = jy = jz = 1                                    ### Model Parameters (Ising interaction Strengths)
sys_size0t = np.linspace(3, L0t, L0t-2).astype(int)
for l in sys_size0t:
    sing_0 = E_0(l, l, t0t, jx, jy, jz)             ### Extracting the singular values
    GE0tt.append(-1 * np.sum(sing_0) / (1 * l*l))   ### Storing the ground state energy for different system sizes in the list
GE0t = np.array(GE0t)

##########################################################################################################################################
################## Relative Ground State Energy in the 0-flux state (Calculations for Twisted Boundary Conditions) #######################
##########################################################################################################################################

rE0t = []
for i, l in enumerate(sys_size0t):
    rE0t.append((GE0t[i]*l*l + l*l * 1.5745974))
rE0t = np.array(rE0t)

##########################################################################################################################################
################# Ground State Energy per unit cell in the 2-flux state (Calculations for Periodic Boundary Conditions) ##################
##########################################################################################################################################

GE2p = []                                           ### Empty List to store the Ground State Energy per unit cell of a 2-flux state in PBC
L2p = 20                                            ### Maximum System Size
jx = jy = jz = 1                                    ### Model Parameters (Ising interaction Strengths)
t2p = 0                                             ### Twist Parameter (= 0 implies PBC)
mu2p = 4                                            ### mu2p is the unit cell number (which gives a coordinate) of a single spin flip
sys_size2p = np.linspace(3, L2p, L2p-2).astype(int)
for l in sys_size2p:
    sing_2 = E_2f(l, l, t2p, jx, jy, jz, mu2p)       ### Extracting the singular values
    GE2p.append(-1 * np.sum(sing_2) / (l * l))       ### Storing the ground state energy for different system sizes in the list
GE2p = np.array(GE2p)

##########################################################################################################################################
################## Relative Ground State Energy in the 2-flux state (Calculations for Periodic Boundary Conditions) ######################
##########################################################################################################################################

rE2p = []
for i, l in enumerate(sys_size2p):
    rE2p.append((GE2p[i]*l*l + l*l * 1.5745974))
rE2p = np.array(rE2p)

##########################################################################################################################################
################# Ground State Energy per unit cell in the 2-flux state (Calculations for Twisted Boundary Conditions) ###################
##########################################################################################################################################

GE2t = []                                           ### Empty List to store the Ground State Energy per unit cell of a 2-flux state in PBC
L2t = 20                                            ### Maximum System Size
jx = jy = jz = 1                                    ### Model Parameters (Ising interaction Strengths)
t2t = 0                                             ### Twist Parameter (= 1 implies TBC)
mu2t = 4                                            ### mu2t is the unit cell number (which gives a coordinate) of a single spin flip
sys_size2t = np.linspace(3, L2t, L2t-2).astype(int)
for l in sys_size2t:
    sing_2 = E_2f(l, l, t2t, jx, jy, jz, mu2t)       ### Extracting the singular values
    GE2t.append(-1 * np.sum(sing_2) / (l * l))       ### Storing the ground state energy for different system sizes in the list
GE2t = np.array(GE2t)

##########################################################################################################################################
################## Relative Ground State Energy in the 2-flux state (Calculations for Twisted Boundary Conditions) #######################
##########################################################################################################################################

rE2t = []
for i, l in enumerate(sys_size2t):
    rE2t.append((GE2t[i]*l*l + l*l * 1.5745974))
rE2t = np.array(rE2t)

##########################################################################################################################################
################# Ground State Energy per unit cell in the 0-flux state (PLOTS for Periodic/Twisted Boundary Conditions) #################
##########################################################################################################################################
sysL = sys_size0p
plt.plot(sysL, GE0p, marker = 'o', markersize = 2, linewidth = 0.6, label = r"$0-$flux State : PBC")
plt.plot(sysL, GE0t, marker = 's', markersize = 2, linewidth = 0.6, label = r"$0-$flux State : TBC")
plt.ylabel(r"Ground State Energy per unit cell : $E_0$")
plt.xlabel(r"System Size : $L_1 = L_2 = L $")
plt.title(f"Maximum System Size : $L_1 = L_2 = {np.max(sysL)}$")
plt.grid(True, linestyle = "--", alpha = 0.5)
# plt.axhline(y = -1.5746, color='r', linestyle='--', linewidth = 0.5, label=r"$E_{GS}^0 \approx -1.57460$")
plt.axhline(y = -1.5745974, color='g', linestyle='--', linewidth = 0.5, label=r"$E_{GS}^0 \approx -1.5745974$")
plt.legend()
plt.tight_layout()
# plt.savefig("0-flux_GS_energy_perunitcell_vs_systemsize.pdf", bbox_inches="tight")       #### To save to plot as .pdf (format)
plt.show()

############ Similarly, You Can Plot Everything and see and compare the Results to what Kitaev has given on page 45 of his paper #########
