#----------- PACKAGES Required! ---------------#
import numpy as np
import scipy as sp
import math
from numpy.linalg import eigh, norm, svd       # To compute the Singular Value Decomposition
from matplotlib import pyplot as plt           # Package required to plot
import matplotlib as mpl
from scipy import stats
mpl.rcParams.update({"text.usetex":True})
plt.rcParams['figure.dpi'] = 100
#-------------- END of PACKAGES ---------------#

################# You Can Generate this Data-Set using the codes in " Comparing Kitaev's Result.py " file #######################
############################### rE2p : relative energy of 2-flux state computed using PBC (t = 0) ###############################
#################################################################################################################################

rE2p = np.array([0.45296037, 0.11736864, 0.2748365 , 0.45822456, 0.170667  ,
       0.24550272, 0.42778611, 0.198265  , 0.24072587, 0.40353696,
       0.21383401, 0.24095848, 0.3851505 , 0.22361856, 0.24238141,
       0.37091844, 0.2302819 , 0.244008  , 0.35962227, 0.23509332,
       0.2455618 , 0.35044416, 0.23871875, 0.24696984, 0.34285599,
       0.24155824, 0.24822115, 0.336474  , 0.24383453, 0.24932352,
       0.33104511, 0.24569624, 0.25030425, 0.32635872, 0.24725509,
       0.2511838 , 0.32228469, 0.248592  , 0.25196509, 0.31871952,
       0.24972594, 0.25266736, 0.315576  , 0.25072484, 0.25330603,
       0.312768  , 0.25160079, 0.2539    , 0.31024728, 0.25239136,
       0.25443922, 0.30801708, 0.25310175, 0.25492544, 0.30595833,
       0.25374652, 0.25540097, 0.304128  , 0.25433035, 0.2558182 ,
       0.3024378 , 0.25485312, 0.256204  , 0.30091248, 0.25533432,
       0.25658576, 0.29951451, 0.25578   , 0.25693977, 0.29823552,
       0.25621832, 0.25726248, 0.29705625, 0.25662768, 0.25761505,
       0.29592576, 0.25700438, 0.25792   ])

#################################################################################################################################
############################### rE2t : relative energy of 2-flux state computed using TBC (t = 1) ###############################
#################################################################################################################################

rE2t = np.array([0.36382239, 0.20416624, 0.24305175, 0.29597184, 0.23133145,
       0.24423616, 0.27865296, 0.243536  , 0.24835008, 0.27147888,
       0.24947949, 0.25142096, 0.26791425, 0.25286144, 0.25361773,
       0.26594568, 0.25500679, 0.255224  , 0.26478081, 0.25647644,
       0.25643804, 0.26405568, 0.2575375 , 0.25738024, 0.26359911,
       0.25835152, 0.25812813, 0.263295  , 0.2589895 , 0.25874432,
       0.2631024 , 0.25949888, 0.259259  , 0.26298432, 0.25993203,
       0.25968896, 0.26292006, 0.260288  , 0.26006751, 0.26288892,
       0.26059806, 0.260392  , 0.2628855 , 0.26088164, 0.26068409,
       0.2628864 , 0.26113276, 0.26095   , 0.26290908, 0.2613416 ,
       0.26118082, 0.26296488, 0.26157175, 0.26141696, 0.26300655,
       0.26175284, 0.26159715, 0.263052  , 0.26192119, 0.2617764 ,
       0.2631447 , 0.26210304, 0.26199225, 0.26318952, 0.26224738,
       0.26213456, 0.2632833 , 0.262395  , 0.26233364, 0.2633472 ,
       0.26255983, 0.26246468, 0.26341875, 0.26269248, 0.26259541,
       0.26349804, 0.26287092, 0.262784  ])

#################################################################################################################################
sysL3 = np.linspace(3, 80, 78).astype(int)                    ###### SYSTEM SIZE's array
#################################################################################################################################

plt.figure(figsize=(10, 6))
plt.plot(sysL3[7:], rE2p[7:], marker = 'o', markersize = 2, linewidth = 0.6, label = r"$2-$flux State : PBC ($t = 0$)")
plt.plot(sysL3[7:], rE2t[7:], marker = 'o', markersize = 2, linewidth = 0.6, label = r"$2-$flux State : TBC ($t = 1$)")
plt.ylabel(r"Relative Energy of an Adjacent Vortex Pair above the Ground State Energy")
plt.xlabel(r"System Size : $L_1 = L_2 = L $")
# plt.yticks(np.arange(0.19, 0.42, 0.02))
# plt.xticks(np.arange(10, 84, 5))
plt.title(r"Energy of an Adjacent Vortex Pair (Zoomed plot!) : Maximum System Size : $L_1 = L_2 = 80$")
plt.grid(True, linestyle = "--", alpha = 0.5)
plt.axhline(y = 0.2575, color='b', linestyle='--', linewidth = 0.5, label=r"$E_{GS}^2-E_{GS}^0 \approx 0.25750$")
plt.axhline(y = 0.26265, color='g', linestyle='--', linewidth = 0.5, label=r"$E_{GS}^2-E_{GS}^0 \approx 0.26265$")
# plt.axhline(y = 0.265, color='c', linestyle='--', linewidth = 0.5, label=r"$E_{GS}^2-E_{GS}^0 \approx 0.265$")
plt.axhline(y = 0.3072, color='r', linestyle='--', linewidth = 0.5, label=r"$E_{vortex} \approx 0.30720$")
plt.legend(loc='upper right')
plt.tight_layout()
# plt.savefig("2-flux_GS_adjacent_vs_systemsize.pdf", bbox_inches="tight")
plt.show()

#################################################################################################################################
################################################### Finite Size Scaling Analysis ################################################
#################################################################################################################################

################## Split the data based on index modulo 3:
dataP_3n   = rE2p[48::3]  # indices: 0, 3, 6, ...
dataP_3np1 = rE2p[49::3]  # indices: 1, 4, 7, ...
dataP_3np2 = rE2p[50::3]  # indices: 2, 5, 8, ...
#######################################################
datat_3n   = rE2t[48::3]  # indices: 0, 3, 6, ...
datat_3np1 = rE2t[49::3]  # indices: 1, 4, 7, ...
datat_3np2 = rE2t[50::3]  # indices: 2, 5, 8, ...
#######################################################
Linv_3n = 1/np.arange(51, 81, 3)
Linv_3np1 = 1/np.arange(52, 80, 3)
Linv_3np2 = 1/np.arange(53, 81, 3)

#################################################################################################################################
################################################# PLOT 1 : PBC ##################################################################
#################################################################################################################################

plt.plot(Linv_3n, dataP_3n, 'o', markersize = 4, label = '$L = 3N$')
plt.plot(Linv_3np1, dataP_3np1, 's', markersize = 4, label = '$L = 3N+1$')
plt.plot(Linv_3np2, dataP_3np2, '*', markersize = 4, label = '$L = 3N+2$')
plt.legend()
plt.show()

#################################################################################################################################
################################################# PLOT 2 : TBC ##################################################################
#################################################################################################################################

plt.plot(Linv_3n, datat_3n, 'o', markersize = 4, label = '$L = 3N$')
plt.plot(Linv_3np1, datat_3np1, 's', markersize = 4,  label = '$L = 3N+1$')
plt.plot(Linv_3np2, datat_3np2, '*', markersize = 4, label = '$L = 3N+2$')
plt.legend()
plt.show()

#################################################################################################################################
############################################# PLOT 3 : FSS => PBC ###############################################################
#################################################################################################################################

# Example: these L arrays correspond to the actual L values behind the inverse you already computed.
L_n   = np.arange(51, 81, 3)   # for L=3N
L_np1 = np.arange(52, 80, 3)  # for L=3N+1
L_np2 = np.arange(53, 81, 3)  # for L=3N+2

# Now, since your plots use 1/L, we compute x = 1/L.
x_n   = 1 / L_n
x_np1 = 1 / L_np1
x_np2 = 1 / L_np2

# Fit a second order polynomial: f(x) = a0 + a1*x + a2*x^2.
# Note: np.polyfit returns coefficients in descending powers: [a2, a1, a0].
coeff_n   = np.polyfit(x_n,   dataP_3n,   2)
coeff_np1 = np.polyfit(x_np1, dataP_3np1, 2)
coeff_np2 = np.polyfit(x_np2, dataP_3np2, 2)

# Extrapolated energy at L -> infinity corresponds to x = 0, i.e., f(0) = a0.
# Since np.polyfit returns [a2, a1, a0], we take the last coefficient.
a0_n   = coeff_n[-1]
a0_np1 = coeff_np1[-1]
a0_np2 = coeff_np2[-1]

print("Extrapolated Energy for L = 3N   :", a0_n)
print("Extrapolated Energy for L = 3N+1 :", a0_np1)
print("Extrapolated Energy for L = 3N+2 :", a0_np2)

# Optional: Plotting the data along with the fitted curves.
x_fit = np.linspace(0, max(np.max(x_n), np.max(x_np1), np.max(x_np2)), 200)
y_fit_n   = np.polyval(coeff_n,   x_fit)
y_fit_np1 = np.polyval(coeff_np1, x_fit)
y_fit_np2 = np.polyval(coeff_np2, x_fit)

plt.figure(figsize=(8, 5))
plt.plot(x_n, dataP_3n, 'o', markersize = 3)
plt.plot(x_np1, dataP_3np1, '*', markersize = 3)
plt.plot(x_np2, dataP_3np2, 'o', markersize = 3)
plt.plot(x_fit, y_fit_n,   '-', linewidth = 0.6, label='$L=3N$')
plt.plot(x_fit, y_fit_np1, '--', linewidth = 1, label='$L=3N+1$')
plt.plot(x_fit, y_fit_np2, linestyle='dotted', linewidth = 1, label=' $L=3N+2$')
plt.xlabel("$1 / L$")
plt.ylabel("Relative Energy")
plt.title('Finit-Size Scalling of Nearest-Neighbour Fluxes : (PBC : $t = 0$)')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

#################################################################################################################################
############################################# PLOT 4 : FSS => TBC ###############################################################
#################################################################################################################################

# Example: these L arrays correspond to the actual L values behind the inverse you already computed.
L_n   = np.arange(51, 81, 3)   # for L=3N
L_np1 = np.arange(52, 80, 3)  # for L=3N+1
L_np2 = np.arange(53, 81, 3)  # for L=3N+2

# Now, since your plots use 1/L, we compute x = 1/L.
x_n   = 1 / L_n
x_np1 = 1 / L_np1
x_np2 = 1 / L_np2

# Fit a second order polynomial: f(x) = a0 + a1*x + a2*x^2.
# Note: np.polyfit returns coefficients in descending powers: [a2, a1, a0].
coeff_n   = np.polyfit(x_n,   datat_3n,   2)
coeff_np1 = np.polyfit(x_np1, datat_3np1, 2)
coeff_np2 = np.polyfit(x_np2, datat_3np2, 2)

# Extrapolated energy at L -> infinity corresponds to x = 0, i.e., f(0) = a0.
# Since np.polyfit returns [a2, a1, a0], we take the last coefficient.
a0_n   = coeff_n[-1]
a0_np1 = coeff_np1[-1]
a0_np2 = coeff_np2[-1]

print("Extrapolated Energy for L = 3N   :", a0_n)
print("Extrapolated Energy for L = 3N+1 :", a0_np1)
print("Extrapolated Energy for L = 3N+2 :", a0_np2)

# Optional: Plotting the data along with the fitted curves.
x_fit = np.linspace(0, max(np.max(x_n), np.max(x_np1), np.max(x_np2)), 200)
y_fit_n   = np.polyval(coeff_n,   x_fit)
y_fit_np1 = np.polyval(coeff_np1, x_fit)
y_fit_np2 = np.polyval(coeff_np2, x_fit)

plt.figure(figsize=(6, 4))
plt.plot(x_n, datat_3n, 'o', markersize = 4)
plt.plot(x_np1, datat_3np1, 'o', markersize = 4)
plt.plot(x_np2, datat_3np2, 'o', markersize = 4)
plt.plot(x_fit, y_fit_n,   '-',  linewidth = 1, label='$L=3N$')
plt.plot(x_fit, y_fit_np1, '--', linewidth = 1, label='$L=3N+1$')
plt.plot(x_fit, y_fit_np2, '-', linewidth = 1, label=' $L=3N+2$')
plt.xlabel("$1 / L$")
plt.ylabel("Relative Energy")
plt.title('Finit-Size Scalling of Nearest-Neighbor Fluxes : (TBC : $t = 1$)')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

#################################################################################################################################
###################################### I Have also uploaded the plots for Reference #############################################
