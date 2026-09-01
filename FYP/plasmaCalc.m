%% This function serves to calculate the Ionisation efficiencies of the 
% HEMP/CFT from the results delivered by the HEMP_solver script. How does
% one use electron temperature to get collision/ionisation probability?

function plasmaCalc

% Xenon Properties at STP
gamma = 1.6773; % Ratio of specific heats
R = 8.3144598; % m^3.Pa/(K.mol), Univ. gas constant
rho = 5.761; % kg/m^3, density
l = .02; % m, thruster chamber length
r = .002; % m, cross section radius 
A = 0.001*pi*r^2; % m^2, Area [[ inlet area is max 0.1% of chamber area -----> pure assumption ]]
Mm = 0.131293; % kg/mol, Molecular mass
M = 2.1801714*10^(-25); % Xe_mass (kg);
T = 273.15+25; % K, Standard Temperature 
sccm2kgs = 0.0000000983009;

2.1801714*10^(-25); % Xe_mass (kg)
q = 1.60217662*10^(-19); % Elementary_charge (Coulombs)

% pps = mdot*(sccm2kg_s/M); % Particles per second

v_sound = sqrt(gamma*R*T/Mm)
mdot_max = rho*v_sound*A
sccm_max = mdot_max/sccm2kgs
V = l*pi*r^2
mass_in_chamber = V*rho
particles_in_chamber = mass_in_chamber/M
particles_per_m3 = particles_in_chamber/V