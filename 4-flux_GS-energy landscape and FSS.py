#------ PACKAGES Required! ------#
import numpy as np
import scipy as sp
from scipy.sparse.linalg import svds
import math
from numpy.linalg import eigh, norm, svd       # To compute the Singular Value Decomposition
from matplotlib import pyplot as plt           # Package required to plot
import matplotlib as mpl
from scipy import stats
mpl.rcParams.update({"text.usetex":True})
plt.rcParams['figure.dpi'] = 100
from matplotlib.colors import LinearSegmentedColormap
#-------- END of PACKAGES -------#

###############################################################################################################################
##################################################### Twisted Boundary Conditions #############################################
###############################################################################################################################
########## Function that returns the twisted boundary condition if given the input t, which is the amount of twist ############

def twist_ybc(L1, L2, t):                             # Twist for y-bonds
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


##########################################################################################################################################
############################################## Real Space Hamiltonian for 4 flux state ###################################################
##########################################################################################################################################

def Ham_4_s(L1, L2, t, jx, jy, jz, lx, ly):
    m = L1 * L2                           # Total number of unit cells in the lattice
    N = m * 2                             # Total number of sites, including the sublattice points (each unit cell includes 2 sitess)
    H = np.zeros((N, N))                  # Defining the Null Matrix Beforehand as most of the entries will be 0
    ty = twist_ybc(L1, L2, t)             # Importing the boundary conditions with a twist in y-direction
    coord = []
    
# Defining the Matrix elements according to the gauge choices for different links
    for i in range(0, m):                 # For xx-link
        H[i, i+m] = +jx
        H[i+m, i] = -jx
        coord.append((i, i+m))
    
    for j in range(1, m+1):               # For yy-link
        if (j % L2 != 0):
            H[j, j+m-1] = +jy
            H[j+m-1, j] = -jy
            coord.append((j, j+m-1))
        
    for k in range(L2, m):                # For zz-link
        H[k, k+m-L2] = +jz
        H[k+m-L2, k] = -jz
        coord.append((k, k+m-L2))
    
# Applying the PBC (Periodic Boundary Conditions)
    for b in range(0, L2):                # For Vertical bonds (zz-links) no twist in z-direction for the moment
        H[b, N-L2+b] = +jz
        H[N-L2+b, b] = -jz
        coord.append((b, b+N-L2))

    for y in range(L1):                   # For yy-bonds
        H[ty[y][0], ty[y][1]] = +jy
        H[ty[y][1], ty[y][0]] = -jy
        coord.append((ty[y][0], ty[y][1]))

######### Introducing the Single Spin Filp terms in z-bonds! (in the bulk, Boundary Conditions also included!) ########
    # mu = int((L1+1)/2)              ######## For ODD System sizes #######
    # H[2*mu**2 - 1, 6*mu**2 - 6*mu + 1] = -jz 
    # H[6*mu**2 - 6*mu + 1, 2*mu**2 - 1] = +jz

######### For EVEN SYSTEM Sizes    #############################
    nu = int((m)/2 + L2/2)
    H[nu, nu+m-L2] = -jy
    H[nu+m-L2, nu] = +jy


    # # H[1, m] = -jy 
    # # H[m, 1] = +jy

    if ((lx,ly) in coord):
        H[lx, ly] = -jz * H[lx, ly]
        H[ly, lx] = -jz * H[ly, lx]
  
    return H , coord  

##########################################################################################################################################
##################################################### Real Space Energy Eigenvalue #######################################################
##########################################################################################################################################

def E_4f_s(L1, L2, t, jx, jy, jz, lx, ly):
    N = L1 * L2 * 2 
    H4, coord = Ham_4_s(L1, L2, t, jx, jy, jz, lx, ly)        
    # Determine Nu
    Nu = H4.shape[0] // 2
    # Split the matrix into four N x N blocks
    F = H4[:Nu, :Nu]                                               # Top-left block
    M = H4[:Nu, Nu:]                                               # Top-right block
    MT = (-1) * H4[Nu:, :Nu]                                       # Bottom-left block
    D = (-1) * H4[Nu:, Nu:]                                        # Bottom-right block
    u4, s4, v4T = sp.linalg.svd(M, full_matrices = False)          # SVD of M
    # u4, s4, v4T = svds(M, k=5, which='SM')                         # 4 smallest singular values
    
    return s4

##########################################################################################################################################
################################################# Def. Coordinate function : COORD #######################################################
##########################################################################################################################################

def COORD(L1, L2, t, jx, jy, jz):
    m = L1 * L2
    N = 2 * m
    w , cord = Ham_4_s(L1, L2, t, jx, jy, jz, 0, 0)
    arr = np.array(cord)
    # Separate the bottom L2 rows and the top part
    bottom_rows = arr[-L2:]
    top_rows = arr[:-L2]
    # List to store the new array rows
    new_list = []
    # Insert each bottom row after every group of L2 rows from the top part
    bottom_idx = 0
    for i in range(len(top_rows)):
        new_list.append(top_rows[i])
        # Check if we completed a group of L2 rows
        if (i + 1) % L2 == 0 and bottom_idx < len(bottom_rows):
            new_list.append(bottom_rows[bottom_idx])
            bottom_idx += 1
    
    # Convert the list back to a NumPy array
    result_arr = np.array(new_list)
    # Split the array into blocks
    top_block = result_arr[:m+L2]      # First m+L2 elements
    middle_block = result_arr[m+L2:N] # Next 2*l2 elements (to be interleaved)
    remaining_block = result_arr[N:N+2*L2]  # The rows between middle block and the final L2
    last_block = result_arr[N+2*L2:]     # Last L2 elements remain as-is
    
    # Keep track of which middle rows have been inserted
    used_middle = [False] * len(middle_block)
    
    # List to accumulate the new order
    new_order = []
    
    # Iterate over the top block and interleave matching middle block rows
    for top_row in top_block:
        new_order.append(top_row)
        # Check each middle row (only if not used already)
        for i, middle_row in enumerate(middle_block):
            if not used_middle[i] and top_row[1] == middle_row[1]:
                new_order.append(middle_row)
                used_middle[i] = True
                # Since each top row is matched with at most one middle row, break after insertion
                break
    
    # Create the final array:
    final_arr = np.vstack((np.array(new_order), remaining_block, last_block))

        # Separate the top 18 rows and the last m rows
    top = final_arr[:-m]     # first N rows
    last9 = final_arr[-m:]   # last m rows
    
    # Break the last9 block into three groups of L2 rows each
    middle_groups = [last9[i:i+L2] for i in range(0, len(last9), L2)]
    
    # Break the top block into three groups of 2*L2 rows each
    top_groups = [top[i:i+2*L2] for i in range(0, len(top), 2*L2)]
    
    # Build the new array by interleaving the groups:
    new_order = []
    for top_group, middle_group in zip(top_groups, middle_groups):
        new_order.append(top_group)     # add 2*L2 rows from the top
        new_order.append(middle_group)    # insert L2 rows from last9
    
    # Combine the groups into one array using vstack
    resultf = np.vstack(new_order)
    return resultf


##########################################################################################################################################
################################################# Def. PlotCOORD #########################################################################
##########################################################################################################################################

def PlotCOORD(L1, L2):
    CO = []
    col = np.linspace(0, L1*L2-1, 2*L1*L2-1)
    row = np.linspace(-L2, 0, 2*L2+1)
    for i in col:
        for j in row:
            if ( j != -L2):
                if ( i >= np.abs(j) ):
                    if ( i <= np.abs(j)+ 2*L2-1 ):
                        if ( check_num(j) == "i" and check_num(i) == "i" ):
                            CO.append((i,j))
                        if ( check_num(j) == "f" and check_num(i) == "f" ):
                            if ( (i-np.abs(j)) % 2 == 0 ):
                                CO.append((i,j))
    
    ######################################################################################
    CO = np.array(CO)
    # Get indices that would sort the array by the second column
    sort_ind = np.lexsort((CO[:, 0], -CO[:, 1]))
    # Use the indices to reorder the rows of the array
    COO = CO[sort_ind]
    return COO


################### Generating Datas ###########################

L1 = L2 = 10
t = 1
jx = jy = jz = 1
m = L1 * L2
N = 2 * m
coord = COORD(L1, L2, t, jx, jy, jz)
coord = coord.tolist()
E4ff = []
for (i,j) in coord:
    sing_4 = E_4f_s(L1, L2, t, jx, jy, jz, i, j)
    # E4ff.append( -1 * np.sum(sing_4) / (1 * L1*L2) )
    E4ff.append(-1 * np.sum(sing_4) + L1*L2 * 1.574598)



############## PLOTS #######################
############################################

gs4_coord = PlotCOORD(L1, L2)
# Assuming gs4_coord and E4ff are defined properly
E4gs = {tuple(coord): value for coord, value in zip(gs4_coord, E4ff)}

# ### For 0th position at the center (for even system sizes)
# x_coords = [coord[0]-(L1-1+1*L2/2.0)-0.5 for coord in E4gs.keys()]
# y_coords = [coord[1]+(L2/2.0)-0.5 for coord in E4gs.keys()]

### For 0th position at the center (for odd system sizes)
x_coords = [coord[0]-(L1-1+L2/2.0) for coord in E4gs.keys()]
y_coords = [coord[1]+(L2/2.0) for coord in E4gs.keys()]

### For 0th position at one end
# x_coords = [coord[0] for coord in E4gs.keys()]
# y_coords = [coord[1] for coord in E4gs.keys()]

values = list(E4gs.values())

# Determine min and max values
vmin, vmax = min(values), max(values)

# Create the scatter plot
plt.figure(figsize=(14, 6))
colors = ["red", "yellow", "green", "cyan", "blue", "indigo"]
custom_cmap = LinearSegmentedColormap.from_list("my_cmap", colors, N=256)

scatter = plt.scatter(x_coords, y_coords, c=values, cmap=custom_cmap, s=30, vmin=vmin, vmax=vmax)
cbar = plt.colorbar(scatter, label="Value")

# Set evenly spaced ticks on the color bar
num_ticks = 20  # Adjust this for more or fewer tick marks
ticks = np.linspace(vmin, vmax, num=num_ticks)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{t:.2f}" for t in ticks])  # Format to 2 decimal places

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.yticks(np.arange(np.min(y_coords), np.max(y_coords)+1, 1))
plt.title('2D Color Plot of 4-flux Ground State Energies : $L_1=L_2=9$')
plt.grid(True, linestyle="--", alpha=1)
# # plt.savefig("4-flux_GS_L1=L2=18.pdf", bbox_inches="tight")
plt.tight_layout()
plt.show()


####################################
