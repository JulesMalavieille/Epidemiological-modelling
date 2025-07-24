import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import seaborn as sns
sns.set()

time = np.linspace(0, 1000, 10000)
A = np.linspace(0, 1, 100)
B = np.linspace(0.1, 2.5, 100)
C = np.linspace(0, 100, 100)
a = 0.5 # Phytoplankton growth
b = 0.25 # Grazing rate 
m = 0.25 # Mortality rate of zooplankton
c = 0.8 # Infection rate of phytoplankton
L = 1 # Time before virus kill phytoplankton 

Pi = 1
Fi = 1
Zi = 1

def modPFZ(t, Y):
    p, f, z = Y
    dp = a*p - (c/L)*p*f - b*p*z
    df = (c/L)*p*f - f/L - b*f*z 
    dz = b*(p+f)*z - m*z**2  
    return[dp, df, dz]


def traj(Yini):
    tfinal = 1000  # in days
    pini, fini, zini = Yini
    t = np.linspace(0, tfinal, 10000)
    S = solve_ivp(modPFZ, [0, tfinal], [pini, fini, zini], t_eval=t)
    return S


"""Grazing as a function of 1 parmeter"""
# brouta_param = []
# for i in range(len(A)):
#     a = 0.5
#     b = 0.25
#     m = A[i]
#     c = 1
#     L = 1
#     S = traj([Pi, Fi, Zi])
#     morta_tot = b * S.y[0][-1] * S.y[2][-1]+ b * S.y[1][-1] * S.y[2][-1] + (S.y[1][-1] / L) 
#     brouta = ((b * S.y[0][-1] * S.y[2][-1]+ b * S.y[1][-1] * S.y[2][-1]) / morta_tot) * 100
#     brouta_param.append(brouta)
    
"""Grazing as a function of 2 parmeters"""
# brouta_param = np.zeros((len(A), len(A)))
# for i in range(len(A)):
#     for j in range(len(A)):
#         a = 0.5
#         b = A[i]
#         m = A[j]
#         c = 1
#         L = 1
#         S = traj([Pi, Fi, Zi])
#         morta_tot = b * S.y[0][-1] * S.y[2][-1]+ b * S.y[1][-1] * S.y[2][-1] + (S.y[1][-1] / L) 
#         brouta = ((b * S.y[0][-1] * S.y[2][-1]+ b * S.y[1][-1] * S.y[2][-1]) / morta_tot) * 100
#         brouta_param[i, j] = brouta

"""Phytoplanktonic biomass as a function of 1 parmeter"""
biomasse = []
for i in range(len(A)):
    a = 0.5
    b = 0.25
    m = A[i]
    c = 1
    L = 1
    S = traj([Pi, Fi, Zi])
    biomasse.append(S.y[0][-1])

"""Grazing as a function of time"""
# time = np.linspace(0, 1000, 10000)
# broutage = []
# for i in range(len(S.y[0])):
#     morta_tot = b * S.y[0][i] * S.y[2][i]+ b * S.y[1][i] * S.y[2][i] + (S.y[1][i] / L) 
#     brouta = ((b * S.y[0][i] * S.y[2][i]+ b * S.y[1][i] * S.y[2][i]) / morta_tot) * 100   
#     broutage.append(brouta)

# brout_tot = np.sum(broutage) / len(broutage)

"""Method to find numerically the period of a limit cycle"""
# def periode_lim(Xi, tps_lim):
#     X = Xi[tps_lim:]
#     x1 = min(X)
#     x2 = max(X)
#     x = (x1 + x2)/2
#     print(x1, x2, x)
#     t = 0
#     for i in range(len(X)):
#         if X[i] > x and X[i+1] < x :
#             if t==0:
#                 vali = i
#             t += 1 
#         if t == 2 :
#             valf = i
#             return valf - vali
        
# per = periode_lim(S.y[0], 5000)
# tps_it = 0.1 # en jour
# periode = per * tps_it

# print()
# print("La période du cycle limite est de ", periode, "jours. ")

""" General model dynamic """
# plt.figure(1)
# plt.plot(S.y[0], S.y[1], color="red")
# plt.xlabel("Phyotplancton sain en $mmol-C.m^{-3}$", fontsize=12)
# plt.ylabel("Phytoplancton infecté en $mmol-C.m^{-3}$", fontsize=12)
# plt.title("Dynamique phytoplancton - phytoplancton infecté", fontsize=18)
# plt.grid()

# plt.figure(2)
# plt.plot(S.y[0], S.y[2], color="green", linewidth=2)
# plt.xlabel("Phytoplancton sain en $mmol-C.m^{-3}$", fontsize=12)
# plt.ylabel("Zooplancton en $mmol-C.m^{-3}$", fontsize=12)
# plt.title("Dynamique phytoplancton - zooplancton", fontsize=18)


# plt.figure(3)
# plt.plot(S.y[1], S.y[2], color="orange")
# plt.xlabel("Phytoplancton infecté")
# plt.ylabel("Zooplancton")
# plt.title("Phytoplancton conssomé par le zooplancton")


# fig = plt.figure(4)
# ax = fig.add_subplot(projection="3d")
# ax.plot(S.y[0], S.y[1], S.y[2], color="b", linewidth=2)
# ax.set_xlabel("Phytoplancton sain en $mmol-C.m^{-3}$", fontsize=12)
# ax.set_ylabel("Phytoplancton infecté en $mmol-C.m^{-3}$", fontsize=12)
# ax.set_zlabel("Zooplancton en $mmol-C.m^{-3}$", fontsize=12)
# plt.title("Dynamique global du modèle PFZ", fontsize=18)


""" Grazing as a function of time """
# plt.figure(5)
# plt.plot(time, broutage)
# plt.xlabel("Time")
# plt.ylabel("Taux de broutage")

""" Grazing as a function of 1 parameter"""
# plt.figure(6)
# plt.plot(A, brouta_param, c="palevioletred", linewidth=2.5)
# plt.ylabel("Broutage sur la mortalité totale en %", fontsize=12)
# plt.xlabel("Valeur de a en $j^{-1}$", fontsize=12)
# plt.title("Fraction du broutage dans la mortalité totale en fonction de la croissance du phytoplancton (a)", fontsize=18)

""" Grazing as a function of 2 parameters """
# plt.figure(7)
# plt.pcolor(A, A, brouta_param, cmap="viridis")
# plt.colorbar(label="broutage sur la mortalité totale en %")
# plt.xlabel("broutage du phytoplancton en (mmol-C.m^-3)^-1.j^-1 (b)")
# plt.ylabel("mortalité du zooplancton en (mmol-C.m^-3)^-1.j^-1 (m)")
# plt.title("Fraction du broutage dans la mortalité totale en fonction de b et m", fontsize=15)

"""Biomasse of phytoplancton as a function of 1 parameter"""
plt.figure(8)
plt.plot(A, biomasse, color="forestgreen", linewidth=2.5)
plt.xlabel("Mortality rate of zooplancton (m) $(mmol-C.m^{-3})^{-1}.j^{-1}$", fontsize=15) 
plt.ylabel("Phytoplancton biomass ($mmol-C.m^{-3}$)", fontsize=15)
plt.title("Biomass of phytoplancton as a fonction of m", fontsize=20)

""" General return """
S = traj([Pi, Fi, Zi])

p = S.y[0][-1]
f = S.y[1][-1]
z = S.y[2][-1]

print()
print("Phytoplankton : ", p)
print("Infected phytoplankton :", f)
print("Zooplankton : ", z)





