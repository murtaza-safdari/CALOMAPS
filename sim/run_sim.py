from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import GeV, mm, deg

SIM = DD4hepSimulation()

# ==========================================
# PARTICLE GUN CONFIGURATION
# ==========================================
# SIM.enableGun = True

# Common particle choices:
#   "gamma"               - photons (default; pure EM showers)
#   "pi+", "pi-", "proton", "neutron"  - hadrons (wider, longer showers)
# SIM.gun.particle = "gamma" 

# Set the energy spectrum using momentum parameters
# SIM.gun.distribution = "uniform"
# SIM.gun.momentumMin = 5.0 * GeV   # Changed from energyMin
# SIM.gun.momentumMax = 100.0 * GeV # Changed from energyMax
# SIM.gun.energy = 50.0 * GeV

# Aim radially outward to hit the ECal Barrel directly
# SIM.gun.position = (0, 0, 0)
# Shoot toward the +Y face (the "top" of the barrel)
# SIM.gun.direction = (0, 1, 0) 
# OR if you want it to hit a specific Z-slice:
# SIM.gun.position = (0, 0, 100) # Offset slightly in Z to avoid the central 'crack'

SIM.enableGun = True
SIM.gun.particle = "gamma" 
# SIM.gun.energy = 50.0 * GeV
SIM.gun.momentumMin = 5.0 * GeV   
SIM.gun.momentumMax = 400.0 * GeV
SIM.gun.position = (0, 0, 0)
# SIM.gun.position = (0, 0, 100 * mm) # Shifted 100mm in Z
# SIM.gun.direction = (0, 1, 0)

# FORCE PENCIL BEAM ALONG +Y AXIS (Theta=90, Phi=90)
SIM.gun.distribution = "uniform"
SIM.gun.phiMin = 90 * deg
SIM.gun.phiMax = 90 * deg
# SIM.gun.phiMin = 75 * deg
# SIM.gun.phiMax = 75 * deg
SIM.gun.thetaMin = 90 * deg
SIM.gun.thetaMax = 90 * deg

# ==========================================
# PHYSICS & TRACKING
# ==========================================
# Standard physics list for calorimeter performance
SIM.physicsList = "FTFP_BERT" 

# ==========================================
# OUTPUT
# ==========================================
SIM.outputFile = "sim_data_gamma_5_100GeV.slcio"