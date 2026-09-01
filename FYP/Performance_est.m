%% A.Muffatti 12/05/2016 - The initiation file - the one file that runs them all... :D
% Performance_est calculates the thrust over a range of mass flow rates and
% outputs the results in a graphical format.

% The input parameters for efficiency will rely on experimental and
% calculated efficiencies

function [ output, g ] = Performance_est(x,gen,id)

global IE CE CI CT p0 phi0 T0 homefolder; % global variables are always a bad idea

homefolder = pwd;

%% PUT INITIAL CALC FOR MASS FLOW RATE HERE

% mass flow rate --> x(3)

% n_b --> ionisation efficiency

%%
% Static Inputs
IE = 12.1; % eV
CE = 0.25;
CI = 0.07;
CT = 0.68; % CE+CI+CT=1
p0 = 0.002;
phi0 = 0; % Background potential
T0 = 0; % cathode electron temp presumably // probably not correct calc this from (Kornfeld G. 2007)'s results

% Input Power
Ua = x(1); % Volts
Ia = x(2); % Amps
mdot = x(3); % Mass flow rate [sccm]

M = 2.1801714*10^(-25); % Xe_mass (kg)
e = 1.60217662*10^(-19); % Elementary_charge (Coulombs)
sccm2kg_s = 0.0000000983009; %
g0 = 9.80655;
pps = mdot*(sccm2kg_s/M); % Particles per second

Nm = (Ia/e)/(pps);
if Nm > 1.2;
    disp('No solution')
    g = -120;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
end

% A question remains... How should the errors be classified? which ones
% aare more important than others? which ones should be schelduled higher?
% /are more critical to the design?

% g = -100 for geometric constraints
% g = -50 for non-converging solver
% g = -20 for excessive Isp
% g = -200 for greater than 100% Efficiency
% g = -70 for non-real solutions
% g = -30 for negative thrust
% g = -35 for negative Isp

%% Geometric Constraints
g = 1;
[g,output] = geoconst(x,gen,id);
if g < 0;
    return
end
if gen == 35 && id == 39;
    disp('No solution')
    g = -1;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
end
% 
%% RUN SOLVER
disp('RUNNING...')
FEMMrun(x,gen,id);
[sol,flag] = HEMP_solver(x,gen,id);
disp(flag)

try mkdir([homefolder '\Outputs\CFTsolver-output']);catch;end
csvwrite([homefolder '\Outputs\CFTsolver-output\CFTsolverOut_gen_' num2str(gen) '_id_' num2str(id) '.dat'],sol);


if flag==4;
    disp('No solution')
    g = -50;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
elseif flag==0;
    disp('No solution')
    g = -51;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
elseif flag==-1;
    disp('No solution')
    g = -52;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
elseif flag==-2;
    disp('No solution')
    g = -53;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
end
%
%% Solver
sol = sol';
BeamPower = sol(1,1); % beam power
% util_eff = (Ia/M)/(pps/e);

n_g =  1 - (sol(6,1)/sol(7,1)); % Grid efficiency
n_b =  BeamPower/(Ua*Ia); % Beam efficiency (thermal)
n_t = n_b*n_g*Nm; % total efficiency
Thrust = sqrt(2.*(mdot*sccm2kg_s).*Ia*Ua.*n_t); % Efficiency term here - 'n_grid' potential across the exit cusp
Isp = Thrust/(mdot*sccm2kg_s*g0);
Pa = Ua*Ia; % Anode Power
%
%% Check solutions
[g, output] = solconst(gen,id,Isp,Thrust,n_t,n_b,Pa);

if g < 0;
    return
end
%
%% Successful Output
g = 1;
disp(['Success  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);

output.f1 = -Thrust*1000; % thrust [mN]
output.f2 = -n_t*100; % total efficiency [%]
output.f3 = -Isp; %Specific Impulse [s]
output.f4 = Pa; % Anode Power [W]
end

%% Geometric constraints
function [g,output] = geoconst(x,gen,id)

% A question remains... How should the errors be classified? which ones
% aare more important than others? which ones should be schelduled higher?
% /are more critical to the design?

% g = -100 for geometric constraints
% g = -50 for non-converging solver
% g = -20 for excessive Isp
% g = -200 for greater than 100% Efficiency
% g = -70 for non-real solutions
% g = -30 for negative thrust
% g = -35 for negative Isp

g = 1;

output.f1 = 0;
output.f2 = 0;
output.f3 = 0;
output.f4 = 0;

if x(4) <= 2.5;
    disp('x(4) <= 2.5')
    g = -100;
    %   g = x(4)-2.5;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
elseif x(4)+.01 >= x(5);
    disp('x(4) >= x(5)')
    g = -101;
    %   g = x(5)-x(4)-1 -10;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
elseif x(5)+.01 >= x(6);
    disp('x(5) >= x(6)')
    g = -102;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
elseif x(6)+.01 >= x(7);
    disp('x(6) >= x(7)')
    g = -103;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
elseif x(7)+.01 >= x(8); % inner sheild radius
    disp('x(7) >= x(8)')
    g = -104;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
elseif x(8)+.01 >= 50; % outer shield radius
    disp('x(8) >= x(9)')
    g = -105;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
end
end

%% Solution Constraints
function [g,output] = solconst(gen,id,Isp,Thrust,n_t,n_b,Pa)

g = 1;
output.f1 = 0;
output.f2 = 0;
output.f3 = 0;
output.f4 = 0;

if Isp >= 10001; % Disregard Excessive Isp - temporary solution needs to be curbed with plasma density calculation
    disp('Isp Excessive')
    g = -20;
    %   g = x(4)-2.5;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
end
if Isp <= 0; % Disregard negative Isp
    disp('Isp Negative')
    g = -35;
    %   g = x(4)-2.5;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
end
if Thrust <= 0; % Disregard Negative Thrust
    disp('Negative Thrust')
    g = -30;
    %   g = x(4)-2.5;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
end
if n_t > 1;
    disp('n_t > 1 :(')
    g = -200;
    %   g = x(4)-2.5;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
end
if n_t < 0;
    disp('n_t < 0 :(')
    g = -201;
    %   g = x(4)-2.5;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
end
if ~isreal(n_b); % determine if solution is complex
    
    disp('n_b is Complex')
    g = -70;
    %   g = x(4)-2.5;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
end
if ~isreal(Thrust); % determine if solution is complex
    disp('Thrust is complex')
    g = -71;
    %   g = x(4)-2.5;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
end
if Pa < 0; % Negative Anode Power
    disp('Negative Anode Power')
    g = -250;
    %   g = x(4)-2.5;
    output.f1 = 0;
    output.f2 = 0;
    output.f3 = 0;
    output.f4 = 0;
    disp(['Fail  < Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
    return
end
end
