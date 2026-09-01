%% HEMP_solver calls the routine to solve the equation system and checks - A.Muffatti 27/04/2016
% the output against the boundary conditions

function [results,Flag,Lbc,Ubc] = HEMP_solver(x,gen,id,Ua,Ia) %(IE,CE,CI,CT,p0,p1,p2,p3,p4,Ua,Ia,phi0,T0)
% % predetermined input values - Shall be replaced with experimental data
% % upon verification of code

% clear; clc % clear the workspace

global p1 p2 p3 p4 % To Access the variables in any script (probably a bad idea)

Ua = x(1);
Ia = x(2);
%x = zeros(30,1);

% IE = 12.1; % eV
% CE = 0.25;
% CI = 0.07;
% CT = 0.67; % CE+CI+CT=1
% 
% % Cusp arrival probabilities from (Kornfeld G. 2007): DM9.2 4-Stage
% p0 = 0.002;
% % p1 = 0.06;
% % p2 = 0.119;
% p3 = 0.160;
% p4 = 0.254;
% Cusp arrival probabilities from FEMM
[p1,p2,p3,p4] = cusp_prob(x,gen,id);

% Input Power
% Ua = 1000; % Volts
% Ia = 1; % Amps
% phi0 = 0; % Background potential
% T0 = 0; % cathode electron temp // probably not correct calc this from (Kornfeld G. 2007)'s results

% vars legend:
% PB = x(1) IL = x(2) EL = x(3) CL = x(4) AL = x(5) phi1 = x(6) phi2 = x(7)
% phi3 = x(8) phi4 = x(9) T1 = x(10) T2 = x(11) T3 = x(12) T4 = x(13)
% I1 = x(14) I2 = x(15) I3 = x(16) I4 = x(17) je0 = x(18) je1 = x(19)
% je2 = x(20) je3 = x(21) je4 = x(22) ji0 = x(23) ji1 = x(24) ji2 = x(25)
% ji3 = x(26) ji4 = x(27) jic1 = x(28) jic2 = x(29) jic3 = x(30)
% phic1 = x(31) ---\
% phic2 = x(32) ----> these 3 are not included
% phic3 = x(33) ---/

% %% Boundary Conditions <<--- doesn't work sukkah
[Lb,Ub,~] = boundaries;

%% Initial estimate + solver
x0 = [Ua*Ia*.8, Ua*Ia*.05, Ua*Ia*.5, Ua*Ia*.5, Ua*Ia*.5, Ua*.5, Ua*.9, Ua*.9, Ua*.9, Ua*Ia*.05,... % Initial guess
    Ua*Ia*.05, Ua*Ia*.05, Ua*Ia*.05, Ia*.5, Ia*.5, Ia*.5, Ia*.5, Ia*.5, Ia*.5, Ia*.5,...
    Ia*.5, Ia*.5, Ia*.5, Ia*.5, Ia*.5, Ia*.5, Ia*.5, Ia*.5, Ia*.5, Ia*.5]; % 0 800 900]; % <-- the last three are the cusp potential x0 terms

% x0fmin = [800, 50, 50, 50, 50, 50, 900, 900, 900, 50, 50, 50, 50,...  % Initial guess
%     1, 1, 1, 1, .2, .2, .2, .5, .8, .2, .2, .2,...
%     .5, .8, .01, .01, .01];

options = optimoptions('lsqnonlin','Display','iter');
options.MaxFunEvals = 10000*length(x0);
options.MaxIter = 2000;
options.TolFun = 1e-50;
% options = optimoptions('fminsearch','Display','iter');

[solution1,~,~,Exitflag] = lsqnonlin(@Power_B_EQs,x0,Lb,Ub,options);
% solution2 = lsqnonlin(@Power_B_EQs,x0,[],[],options)';
% solution3 = fsolve(@Power_B_EQs,x0)';
% solution4 = fminsearch(@Power_B_EQs_fmin,x0fmin)';


%   [X,RESNORM,RESIDUAL,EXITFLAG] = LSQNONLIN(FUN,X0,...) returns an
%   EXITFLAG that describes the exit condition. Possible values of EXITFLAG
%   and the corresponding exit conditions are listed below. See the
%   documentation for a complete description.
%
%     1  LSQNONLIN converged to a solution.
%     2  Change in X too small.
%     3  Change in RESNORM too small.
%     4  Computed search direction too small.
%     0  Too many function evaluations or iterations.
%    -1  Stopped by output/plot function.
%    -2  Bounds are inconsistent.

x = solution1; %,solution2,solution3]; %,solution4];
%% BC Check
% Solution1
% Lbc(1,:) = [   % Lower Bound
%     x(10,1)>0;    % Electron temperatures
%     x(11,1)>0;
%     x(12,1)>0;
%     x(13,1)>0;
%     x(6,1)>0; % Cell Potential
%     x(7,1)>x(6,1);
%     x(8,1)>x(7,1);
%     x(9,1)>=Ua;
%     abs(x(26,1))>abs(x(27,1))]'; % Ion current
% % Upper Bound
% Ubc(1,:) = [
%     x(6,1)<x(7,1);  % Potential balance (cells)
%     x(7,1)<x(8,1);
%     x(8,1)<x(9,1);
%     x(9,1)<1.5*Ua;
%     x(27,1)<0]';
% % % Solution2
% Lbc(2,:) = [   % Lower Bound
%     x(10,2)>0;    % Electron temperatures
%     x(11,2)>0;
%     x(12,2)>0;
%     x(13,2)>0;
%     x(6,2)>0; % Cell Potential
%     x(7,2)>x(6,2);
%     x(8,2)>x(7,2);
%     x(9,2)>Ua;
%     abs(x(26,2))>abs(x(27,2))]'; % Ion current
% % Upper Bound
% Ubc(2,:) = [
%     x(6,2)<x(7,2);  % Potential balance (cells)
%     x(7,2)<x(8,2);
%     x(8,2)<x(9,2);
%     x(9,2)<1.5*Ua;
%     x(27,2)<0]';
% % Solution3
% Lbc(3,:) = [   % Lower Bound
%     x(10,3)>0;    % Electron temperatures
%     x(11,3)>0;
%     x(12,3)>0;
%     x(13,3)>0;
%     x(6,3)>0; % Cell Potential
%     x(7,3)>x(6,3);
%     x(8,3)>x(7,3);
%     x(9,3)>Ua;
%     abs(x(26,3))>abs(x(27,3))]'; % Ion current
% % Upper Bound
% Ubc(3,:) = [
%     x(6,3)<x(7,3);  % Potential balance (cells)
%     x(7,3)<x(8,3);
%     x(8,3)<x(9,3);
%     x(9,3)<1.5*Ua;
%     x(27,3)<0]';

results = solution1; % Lbtarget' Lb' Ub' x0'];
Flag = Exitflag;
end






