from pysmt.shortcuts import *
from wmipa import WMI
from wmipa.integration.volesti_integrator import VolestiIntegrator


from fairsquare_pywmi import convert

from sys import argv

unfair_path = 'example_unfair.fr'
fair_path = 'example_fair.fr'

epsilon = 0.1

unfair = convert(unfair_path)
fair = convert(fair_path)

for prog in [unfair, fair]:

    wmi = WMI(prog.support, prog.weight, integrator=VolestiIntegrator)
    M, MH, nMH = prog.queries
    p_M, _ = wmi.computeWMI(M, mode=WMI.MODE_SA_PA_SK)
    p_MH, _ = wmi.computeWMI(MH, mode=WMI.MODE_SA_PA_SK)
    p_nMH, _ = wmi.computeWMI(nMH, mode=WMI.MODE_SA_PA_SK)

    p_H_g_M = p_MH / p_M
    p_H_g_nM = p_nMH / (1 - p_M)

    print("Pr(M):", p_M)
    print("Pr(H|M):", p_H_g_M)
    print("Pr(H|~M):", p_H_g_nM)

    ratio = p_H_g_M / p_H_g_nM

    eps_fairness = (ratio > (1 - epsilon))

    print("Ratio:", ratio, f"{epsilon}-fairness:", eps_fairness,'\n')

