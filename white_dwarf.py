"""
This is a copy of the Jupyter notebook, in .py form and as a function so you can play with the inputs.
"""

import bamboo as bam
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import pypropep as ppp
from CoolProp.CoolProp import PropsSI

# STARTING PARAMETERS
# Combustion operating conditions
pc = 10e5                   # Chamber pressure (Pa)
thrust = 1.5e3              # Desired thrust (N)
p_amb = 1.01325e5           # Ambient pressure (Pa). 1.01325e5 is sea level atmospheric.
OF_ratio = 3.5              # Oxidiser/fuel mass ratio
water_mass_fraction = 0.20  # Fraction of the fuel that is water, by mass

# Chamber geometry
inner_wall_thickness = 1.5e-3
Dc = 120.5e-3                  
L_star = 1.2           
copper_material = bam.materials.CopperC106   
graphite_material = bam.materials.Graphite

# Coolant jacket
mdot_coolant = 1                                      # Coolant mass flow rate (kg/s) - 1 kg/s ~= 60 litre/min for water
inlet_T = 298.15                                      # Coolant inlet temperature (K)
inlet_p0 = 1.01325e5                                  # Tank / inlet coolant stagnation pressure (Pa)
ideal_channel_height = 1.45e-3
number_of_fins = 100
ideal_blockage_ratio = 0.4                            # Fraction of area that is blocked by 'fins' - ignoring copper-stainless gap
copper_stainless_gap = 0.5e-3                         # Gap between fin tips and stainless ID

# Convert all inputs to dictionary form
inputs = {"pc" : pc,
          "thrust" : thrust,
          "p_amb" : p_amb,
          "OF_ratio" : OF_ratio,
          "water_mass_fraction" : water_mass_fraction,
          "Dc" : Dc,
          "L_star" : L_star,
          "copper_material" : copper_material,
          "graphite_material" : graphite_material,
          "mdot_coolant" : mdot_coolant,
          "inlet_T" : inlet_T,
          "inlet_p0" : inlet_p0,
          "number_of_fins" : number_of_fins,
          "ideal_blockage_ratio" : ideal_blockage_ratio,
          "copper_stainless_gap" : copper_stainless_gap}

# SIMULATION 
def get_data(inputs = inputs):
    pc = inputs["pc"]
    thrust = inputs["thrust"]
    p_amb = inputs["p_amb"]
    OF_ratio = inputs["OF_ratio"]
    water_mass_fraction = inputs["water_mass_fraction"]
    L_star = inputs["L_star"]
    copper_material = inputs["copper_material"]
    graphite_material = inputs["graphite_material"]
    mdot_coolant = inputs["mdot_coolant"]
    inlet_T = inputs["inlet_T"]
    inlet_p0 = inputs["inlet_p0"]
    number_of_fins = inputs["number_of_fins"]
    ideal_blockage_ratio = inputs["ideal_blockage_ratio"]
    copper_stainless_gap = inputs["copper_stainless_gap"]

    # Initialise both a frozen-flow and equilibrium-flow model
    ppp.init()
    p_froz = ppp.FrozenPerformance()
    p_shift = ppp.ShiftingPerformance()

    # Propellant set up
    ipa = ppp.PROPELLANTS['ISOPROPYL ALCOHOL']
    water = ppp.PROPELLANTS['WATER']
    n2o = ppp.PROPELLANTS['NITROUS OXIDE']

    # Add propellants
    ipa_mass = (1 - water_mass_fraction)
    water_mass = water_mass_fraction
    n2o_mass = OF_ratio

    p_froz.add_propellants_by_mass([(ipa, ipa_mass), (water, water_mass), (n2o, n2o_mass)])
    p_shift.add_propellants_by_mass([(ipa, ipa_mass), (water, water_mass), (n2o, n2o_mass)])
                    
    # Set chamber pressure and exit pressure
    p_froz.set_state(P = pc/1e5, Pe = p_amb/1e5)                      
    p_shift.set_state(P = pc/1e5, Pe = p_amb/1e5)                      

    # Retrieve perfect gas properties (index 0 means chamber conditions, 1 means throat)
    gamma = p_froz.properties[1].Isex   # 'Isex' is the ratio of specific heats
    cp = 1000*p_froz.properties[1].Cp   # Cp is given in kJ/kg/K, we want J/kg/K
    Tc = p_froz.properties[0].T

    # Get specific impulse
    isp_froz = p_froz.performance.Isp
    isp_shift = p_shift.performance.Isp

    mdot_froz = thrust/isp_froz        
    mdot_shift = thrust/isp_shift

    # Get throat area and area ratio from pypropep
    At_froz = p_froz._equil_structs[1].performance.a_dotm * mdot_froz / 101325  # pypropep uses units of atm, need to convert to Pa
    At_shift = p_shift._equil_structs[1].performance.a_dotm * mdot_shift / 101325

    area_ratio_froz = p_froz.performance.ae_at 
    area_ratio_shift = p_shift.performance.ae_at 

    Ae_froz = At_froz * area_ratio_froz
    Ae_shift = At_shift * area_ratio_shift

    # Useful intermediate values
    Rc = Dc / 2
    Dt = 2*(At_froz/np.pi)**0.5
    Ac = np.pi * Rc**2   
    Vc = L_star * At_froz         # From the definition of L*

    # Equation (1.34) from http://www.braeunig.us/space/propuls.htm 
    Lc = ( 24*Vc/np.pi - (Dc**3 - Dt**3)/np.tan(45*np.pi/180) ) / (6*Dc**2)   

    # Rao bell nozzle geometry
    xs, ys = bam.rao.get_rao_contour(r_c = Rc, 
                                    r_t = Dt / 2, 
                                    area_ratio = area_ratio_froz, 
                                    L_c = Lc, 
                                    theta_conv = 45)

    # Graphite extends from the nozzle contour to the inside of the copper tube
    def graphite_thickness(x):
        y = np.interp(x, xs, ys)
        return Rc - y

    # Correct the blockage ratio and channel height, to account for the copper-stainless gap
    blocked_area = ( np.pi*(Rc + inner_wall_thickness + ideal_channel_height)**2 - np.pi*(Rc + inner_wall_thickness)**2) * ideal_blockage_ratio
    total_area = np.pi*(Rc + inner_wall_thickness + ideal_channel_height + copper_stainless_gap)**2 - np.pi*(Rc + inner_wall_thickness)**2
    blockage_ratio = blocked_area / total_area       
    channel_height = ideal_channel_height + copper_stainless_gap

    # Setup the exhaust gas transport properties using Cantera
    gri30 = ct.Solution('gri30.yaml')

    ipa_ct = ct.Quantity(gri30, constant = "TP", mass = ipa_mass)
    ipa_ct.TPX = Tc, pc, "C:3, H:8, O:1"

    water_ct = ct.Quantity(gri30, constant = "TP", mass = water_mass)
    water_ct.TPX = Tc, pc, "H:2, O:1"

    water_ct = ct.Quantity(gri30, constant = "TP", mass = water_mass)
    water_ct.TPX = Tc, pc, "H:2, O:1"

    n2o_ct = ct.Quantity(gri30, constant = "TP", mass = n2o_mass)
    n2o_ct.TPX = Tc, pc, "N:2, O:1"

    # Mix and reach equilibrium at chamber conditions
    quantity = ipa_ct + water_ct + n2o_ct
    quantity.equilibrate("TP")

    gas = ct.Solution('gri30.yaml')
    gas.TPY = quantity.TPY

    def mu_exhaust(T, p):
        gas.TP = T, p
        return gas.viscosity

    def k_exhaust(T, p):
        gas.TP = T, p
        return gas.thermal_conductivity

    def Pr_exhaust(T, p):
        gas.TP = T, p
        return gas.cp * gas.viscosity / gas.thermal_conductivity # Definition of Prandtl number


    # Coolant is pure water - use CoolProp
    def Pr_coolant(T, p):
        try:
            return PropsSI("PRANDTL", "T", T, "P", p, "WATER")
        except ValueError:
            return PropsSI("PRANDTL", "Q", 0, "P", p, "WATER")

    def mu_coolant(T, p):
        try:
            return PropsSI("VISCOSITY", "T", T, "P", p, "WATER")
        except ValueError:
            return PropsSI("VISCOSITY", "Q", 0, "P", p, "WATER")

    def k_coolant(T, p):
        try:
            return PropsSI("CONDUCTIVITY", "T", T, "P", p, "WATER")
        except ValueError:
            return PropsSI("CONDUCTIVITY", "Q", 0, "P", p, "WATER")

    def rho_coolant(T, p):
        try:
            return PropsSI("DMASS", "T", T, "P", p, "WATER")
        except ValueError:
            return PropsSI("DMASS", "Q", 0, "P", p, "WATER")

    def cp_coolant(T, p):
        try:
            return PropsSI("CPMASS", "T", T, "P", p, "WATER")
        except ValueError:
            return PropsSI("CPMASS", "Q", 0, "P", p, "WATER")


    # Transport property objects
    exhaust_transport = bam.TransportProperties(Pr = Pr_exhaust,
                                                mu = mu_exhaust,
                                                k = k_exhaust)

    coolant_transport = bam.TransportProperties(Pr = Pr_coolant,
                                                mu = mu_coolant,
                                                k = k_coolant,
                                                cp = cp_coolant,
                                                rho = rho_coolant)

    # Set up the objects we need
    perfect_gas = bam.PerfectGas(gamma = gamma, cp = cp)                   # Approximate values for CO2
    chamber_conditions = bam.ChamberConditions(p0 = pc, T0 = Tc)
    geometry = bam.Geometry(xs = xs, rs = ys)

    copper_wall = bam.Wall(material = copper_material, thickness = inner_wall_thickness) 
    graphite_wall = bam.Wall(material = graphite_material, thickness = graphite_thickness) 

    # Main inputs
    engine = bam.Engine(perfect_gas = perfect_gas, 
                        chamber_conditions = chamber_conditions, 
                        geometry = geometry,
                        exhaust_transport = exhaust_transport,
                        walls = [graphite_wall, copper_wall])

    # Add a cooling jacket to the engine
    engine.cooling_jacket = bam.CoolingJacket(T_coolant_in = inlet_T, 
                                            p_coolant_in = inlet_p0, 
                                            mdot_coolant = mdot_coolant, 
                                            channel_height = channel_height,
                                            blockage_ratio = blockage_ratio,
                                            number_of_channels = number_of_fins,
                                            coolant_transport = coolant_transport,
                                            configuration = 'vertical',
                                            restrain_fins = False)

    cooling_data = engine.steady_heating_analysis()

    return cooling_data