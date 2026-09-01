%% NOTES ON SCRIPT - A.Muffatti 27/04/2016
% Make sure this script is robust and define you upper and lower bound
% values for efficiency and thrust - from experimental results. Ideally the script will always
% converge.


%Script conatians the basic equations for the 1D power balance model of the
%HEMP/CFT taken from;
%% Kornfeld G, Koch N, Harmann HP. Physics and evolution of HEMP-thrusters. InProceedings of the 30th International Electric Propulsion Conference 2007 Sep 17 (pp. 17-20).
% http://erps.spacegrant.org/uploads/images/images/iepc_articledownload_1988-2007/2007index/IEPC-2007-108.pdf

%% LEGEND
% The simplified equation system contains 28 equations to be solved
% simultaneously. 'sub(sequent)' script identifiers indicate what particle,
% property or location in the thruster model.
%
% phi (?) = potential (V)
% I = ionization source current (A)
% T = electron temperatures (eV)
% e = electron property
% i = ion property
% c = cusp property at ceramic surface
% j = current (A)
% Ua = anode potential (V)
% Ia = anode current (A)
% x(1) = Beam Power
% x(2) = Ionization Loss
% x(3) = Excitation Loss
% x(4) = Cusp Loss
% x(5) = Anode Loss
% p = Cusp Arrival probability
% Numbers following a definition denote the potential cell of which the
% quantity resides

% CE = relative proportion of gained electron power transferred to excitation
% CI = " " " " " " " " ionisation
% CT = " " " " " " " " thermalisation
% Kornfeld gives values of:
% CE = 0.25, CI = 0.07, CT = 0.68   -->   CE+CI+CT=1
% These should later be replace with experimental values upon confirmation
% of the code's operation
% IE = Ionization energy for Xenon
% IE = 12.1 eV

% Unknown Quantities
% - 4 plasma potentials x(6) to x(9),
% - 3 cusp potentials x(31) to x(33) at the ceramic surface,
% - 4 electron temperatures x(10) to x(13),
% - 4 ionization source currents x(14) to x(17) in the plasma cells,
% - 5 electron currents x(18) to x(22),
% - 5 ion currents x(23) to x(27),
% - 3 ion cusp currents x(28) to x(30).

%%
function Power_B = Power_B_EQs(x) %,IE,CE,CI,CT,p0,p1,p2,p3,p4,Ua,Ia,phi0,T0)
% % predetermined input values - Shall be replaced with experimental data
% % upon verification of code
% IE = 12.1; % eV
% CE = 0.25;
% CI = 0.07;
% CT = 0.67; % CE+CI+CT=1
%
% % Cusp arrival probabilities from (Kornfeld G. 2007): DM9.2 4-Stage
% p0 = 0.002;
% p1 = 0.06;
% p2 = 0.119;
% p3 = 0.160;
% p4 = 0.254;
% [p1,p2,p3,p4] = cusp_prob; % cusp pro from FEMM

% % Input Power
% Ua = 1000; % Volts
% Ia = 1; % Amps
% phi0 = 0; % Background potential
% T0 = 0; % cathode electron temp // probably not correct calc this from (Kornfeld G. 2007)'s results

global IE CE CI CT p0 p1 p2 p3 p4 Ua Ia phi0 T0 % Access predefined parameters

Power_B = [
    %% Definition equations (15)
    % electron current
    -x(18)+p0*x(6)^(3/2);
    -x(19)+x(18)*(1-p1)+x(14);
    -x(20)+x(19)*(1-p2)+x(15);
    -x(21)+x(20)*(1-p3)+x(16);
    % Ionization source current
    -x(14)+x(18)*(1-p1)*CI*((x(6)-phi0+T0)/IE);
    -x(15)+x(19)*(1-p2)*CI*((x(7)-x(6)+x(10))/IE);
    -x(16)+x(20)*(1-p3)*CI*((x(8)-x(7)+x(11))/IE);
    -x(17)+x(21)*(1-p4)*CI*((x(9)-x(8)+x(12))/IE);
    % ion current
    -x(23) + x(24) + x(14) + x(28);
    -x(24) + x(25) + x(15) + x(29);
    -x(25) + x(26) + x(16) + x(30);
    -x(26) + x(17) + abs(x(27));
    % Electron temperatures
    -x(11)+(CT*x(19)*(1-p2)*(x(7)-x(6)+x(10)))/x(20);
    -x(12)+(CT*x(20)*(1-p3)*(x(8)-x(7)+x(11)))/x(21);
    -x(13)+(CT*x(21)*(1-p4)*(x(9)-x(8)+x(12)))/(x(21)*(1-p4)+x(17));
    %% Current balances and boundary interface conditions (8)
    % Constant current at 3 interfaces and 2 electodes
    -Ia + x(22) + x(27);
    -Ia + x(21) + x(26);
    -Ia + x(20) + x(25);
    -Ia + x(19) + x(24);
    -Ia + x(18) + x(23);
    % Zero current at 3 dielectric cusps
    x(18)*p1 - x(28);
    x(19)*p2 - x(29);
    x(20)*p3 - x(30);
    %% Power Balance equations for each magnetic cell (4)
    % Recieved electron power from downstream cell = direct cusp loss +
    % thermalised power + ionisation loss + excitation loss.
    % % there is also the repitition in these equations where a term is
    % % essentially cancelled (either side of the equalls sign) - x(2)lustration of
    % % the physics or mistake? Are the cusp potentials needed? -> maybe not as
    % they should be in equilibrium....
    
    % % x(18)*(1-p1)*(x(6)-phi0+T0)+x(18)*p1*(x(31)-phi0+T0)-x(18)*p1(x(31)-phi0+T0)-(x(18)*(1-p1)+x(14))*x(10)-x(14)*IE-x(18)*(1-p1)*CE*(x(6)-phi0+T0);
    % % x(19)*(1-p2)*(x(7)-x(6)+x(10))+x(19)*p2*(x(32)-x(6)+x(10))-x(19)*p2(x(32)-x(6)+x(10))-(x(19)*(1-p2)+x(15))*x(11)-x(15)*IE-x(19)*(1-p2)*CE*(x(7)-x(6)+x(10));
    % % x(20)*(1-p3)*(x(8)-x(7)+x(11))+x(20)*p3*(x(33)-x(7)+x(11))-x(20)*p3(x(33)-x(7)+x(11))-(x(20)*(1-p3)+x(16))*x(12)-x(16)*IE-x(20)*(1-p3)*CE*(x(8)-x(7)+x(11));
    % % x(21)*(1-p4)*(x(9)-x(8)+x(12))+x(21)*p4*(phic4-x(8)+x(12))-x(21)*p4(phic4-x(8)+x(12))-(x(21)*(1-p4)+x(17))*x(13)-x(17)*IE-x(21)*(1-p4)*CE*(x(9)-x(8)+x(12));
    x(18)*(1-p1)*(x(6)-phi0+T0)-(x(18)*(1-p1)+x(14))*x(10)-x(14)*IE+x(18)*(1-p1)*CE*(x(6)-phi0+T0); % null terms required to be removed to be indexed
    x(19)*(1-p2)*(x(7)-x(6)+x(10))-(x(19)*(1-p2)+x(15))*x(11)-x(15)*IE+x(19)*(1-p2)*CE*(x(7)-x(6)+x(10));
    x(20)*(1-p3)*(x(8)-x(7)+x(11))-(x(20)*(1-p3)+x(16))*x(12)-x(16)*IE+x(20)*(1-p3)*CE*(x(8)-x(7)+x(11));
    x(21)*(1-p4)*(x(9)-x(8)+x(12))-(x(21)*(1-p4)+x(17))*x(13)-x(17)*IE+x(21)*(1-p4)*CE*(x(9)-x(8)+x(12));    
    
    %% Global Power balance (1)
    -x(1)+x(26)*x(9)+(x(25)-x(26))*x(8)+(x(24)-x(25))*x(7)+(x(23)-x(24))*x(6);
    -x(2)+IE*(x(14)+x(15)+x(16)+x(17));
    -x(3)+CE*(x(18)*(1-p1)*x(6)+x(19)*(1-p2)*(x(7)-x(6)+x(10))+x(20)*(1-p3)*(x(8)-x(7)+x(11))+x(21)*(1-p4)*(x(9)-x(8)+x(12)));
    % %suspect there is a mistake in x(4) below. In each term phic# term is added and
    % %subtracted leaving a zero term --> is this intentional to illustrate the
    % %physics or is this a mistake?
    % -x(4)+x(18)*p1*(x(31)-phi0+x(6)-x(31)+IE)+x(19)*p2*(x(32)-x(6)+x(7)-x(32)+IE+x(10))+x(20)*p3*(x(33)-x(7)+x(8)-x(33)+IE+x(11));
    -x(4)+x(18)*p1*(-phi0+x(6)+IE)+x(19)*p2*(-x(6)+x(7)+IE+x(10))+x(20)*p3*(-x(7)+x(8)+IE+x(11)); % zero term removed for indexing
    -x(5)+x(21)*p4*(Ua-x(8)+x(12))+(x(17)+x(21)*(1-p4))*(x(9)-Ua+x(13))-x(27)*(x(9)-Ua);
    x(1) + x(2) + x(3) + x(4) + x(5) - Ua*Ia;
    
    %% Boundary Conditions - Apply these conditions after the fact to assess solution space..
    %
    % Electron temperatures
    x(10)>=0;
    x(11)>=0;
    x(12)>=0;
    x(13)>=0;
    
    %Potential balance (cusp+cell)
    % x(31)<x(6);
    % x(32)<x(7);
    % x(33)<x(8);
    
    %Potential balance (cusp)
    % 0<x(31);
    % x(31)<x(32);
    % x(32)<x(33);
    
    % Potential balance (cells)
    x(6)>=0;
    x(6)-x(7)<=0;
    x(7)-x(8)<=0;
    x(8)-x(9)<=0;
    x(9)-1000>=0; % this is the anode potential (Ua) and is variable
    
    % Ion current
    abs(x(26))-abs(x(27))>=0;
    x(27)<=0]; % Due to the opposite direction of x(27) to the anode this current must be negative.

%%









end