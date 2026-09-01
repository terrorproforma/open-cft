function [Lb,Ub,Lbtarget] = boundaries
%BOUNDARIES Boundary conditions for each of the parameters

% vars legend:
% PB = x(1) IL = x(2) EL = x(3) CL = x(4) AL = x(5) 
% phi1 = x(6) phi2 = x(7) phi3 = x(8) phi4 = x(9) 
% T1 = x(10) T2 = x(11) T3 = x(12) T4 = x(13)
% I1 = x(14) I2 = x(15) I3 = x(16) I4 = x(17) 
% je0 = x(18) je1 = x(19) je2 = x(20) je3 = x(21) je4 = x(22) 
% ji0 = x(23) ji1 = x(24) ji2 = x(25) ji3 = x(26) ji4 = x(27) 
% jic1 = x(28) jic2 = x(29) jic3 = x(30)


global Ua Ia

% Lower Bound
% % Lb = [0 0 0 0 0 0 phi1 phi2 Ua 0 0 0 0 0 0 0 0 0 ...
% %     0 0 0 0 0 0 0 abs(ji4) 0 0 0 0];
 
Lb = [Ua*Ia*.001 Ua*Ia*.001 Ua*Ia*.001 Ua*Ia*.001 Ua*Ia*.001 Ua*.001 Ua*.5 Ua*.92 Ua Ua*.001 ...
    Ua*.001 Ua*.001 Ua*.001 Ia*.001 Ia*.1 Ia*.1 Ia*.1 Ia*.1 Ia*.1 Ia*.1 ...
    Ia*.1 Ia*.1 Ia*.1 Ia*.1 Ia*.1 Ia*.1 -Ia Ia*.001 Ia*.001 Ia*.001];

%Upper Bound
% % Ub = [Ua Ua Ua Ua Ua phi2 phi3 phi4 Ua*1.5 Ua ...
% %     Ua Ua Ua 1.1 1.1 1.1 1.1 1.1 1.1 1.1 1.1 1.1 1.1 ...
% %     1.1	1.1	1.1	abs(ji3) 1.1 1.1 1.1];

Ub = [Ua*Ia Ua*Ia Ua*Ia Ua*Ia Ua*Ia Ua Ua Ua Ua Ua*Ia ...
    Ua*Ia Ua*Ia Ua*Ia Ia Ia Ia Ia Ia Ia Ia ...
    Ia Ia Ia Ia Ia Ia -Ia*.001 Ia Ia Ia];

%% Forced Boundaries [Konfeld et al.]
Lbtarget = [891.6 12.3 51.43 22.9 27.7 14.1 1000 1000 1000 8.9...
    100.1 43.1 23.5 0.008 0.543 0.31 0.157 0.106 0.107 0.637...
    0.845 1.002 0.894 0.893 0.363 0.155 -0.002 0.007 0.013 0.102];

% Lb = [890 12 51 22 27 14 900 900 1000 8 ...
%     100 43 23 0.001 0.5 0.3 0.1 0.1 0.1 0.6 ...
%     0.8 1 0.8 0.8 0.3 0.1 -0.03 0.001 0.001 0.001];
% 
% Ub = [900 13 52 23 28 15 1000 1000 1001 9 ...
%     101 44 24 0.01 0.6 0.4 0.2 0.2 0.2 0.7 ...
%     0.9 1.2 0.9 0.9 0.4 0.2 -0.001 0.01 0.1 0.2];
end

