%% FEMM Opening script etc.. first you  run it then you parameterizessss it...
function FEMMrun(x,gen,id)
global homefolder
openfemm; %you need to open the program

% % % % B/c FEMM get confused when running multiple instances and needs to
% % % % allocate which window it has to concentrate on..
% % % global HandleToFEMM
% % % % Create empty array
% % % hand = cell(96,1);
% % % hand(id) = {HandleToFEMM};

newdocument(0); %new magnetostatics problem

% Define the problem type.  Magnetostatic; Units of mm; Axisymmetric; 
% Precision of 10^(-8) for the linear solver; a placeholder of 0 for 
% the depth dimension, and an (mesh)angle constraint of 30 degrees

mi_probdef(0, 'millimeters', 'axi', 1.e-8, 0, 30);

%% Geometric Definition

% temporary values - NEED TO BE REPLACED WITH X VALUES

% R
a = 2;
b = x(4); %3;
d = x(5); %9;
c = (b+d)/2;
e = x(6); %11; % inner sheild radius
%iron shielding
r = x(7); % outer sheild radius
s = x(8); 


% Z
f = 4;
g = 5;
h = 15;
k = 16;
l = 20;
p = 21;

mi_addnode(0,p)
%Vertical

% mi_drawline(r1, z1, r2, z2) % #
mi_drawline(a,0,a,p) % 1
mi_drawline(b,0,b,l) % 2
mi_drawline(c,f,c,g) % 5
mi_drawline(c,h,c,k) % 6
mi_drawline(d,0,d,f) % 7
mi_drawline(d,g,d,h) % 8
mi_drawline(d,k,d,l) % 9
mi_drawline(e,0,e,p) % 10
mi_drawline(c,l,c,p) % 11

mi_drawline(r,0,r,p)
mi_drawline(s,0,s,p)



%Horizontal
mi_drawline(b,f,d,f); % 1
mi_drawline(b,g,d,g); % 2
mi_drawline(b,h,d,h); % 3
mi_drawline(b,k,d,k); % 4
mi_drawline(b,l,d,l); % 5
mi_drawline(a,p,e,p); % 6
mi_drawline(0,0,e,0); % 7

mi_drawline(e,0,r,0)
mi_drawline(e,p,r,p)
mi_drawline(r,0,s,0)
mi_drawline(r,p,s,p)




%% Solution Boundary
% Draw a half-circle to use as the outer boundary for the problem

mi_drawarc([0 -60; 0 80], 180, 2.5);
mi_addsegment([0 -60; 0 80]);

% Define an "asymptotic boundary condition" property.  This will mimic
% an "open" solution domain

muo = pi*4.e-7;

mi_addboundprop('Asymptotic', 0, 0, 0, 0, 0, 0, 1/(muo*0.2), 0, 2);

% Apply the "Asymptotic" boundary condition to the arc defining the
% boundary of the solution region

mi_selectarcsegment(70,0);
mi_setarcsegmentprop(2.5, 'Asymptotic', 0, 0);

%% Add Materials

mi_getmaterial('SmCo 27 MGOe'); % Magnet material
mi_getmaterial('Pure Iron'); % Inner spacer Material
mi_getmaterial('Aluminum, 6061-T6'); % Outer sheilding material
mi_getmaterial('Air'); % Vacuum (free space)

mi_addmaterial('BN Ceramic', 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0); % Chamber wall material, Same magnetic properties as free space

%% Block Labels

% magnets
mi_addblocklabel((b+d)/2,f/2); % magnet #1 starting from anode
mi_addblocklabel((b+d)/2,(g+h)/2); % #2
mi_addblocklabel((b+d)/2,(k+l)/2); % #3

% Outer sheilding / thruster housing material
mi_addblocklabel((d+e)/2,p/3);
mi_addblocklabel((r+s)/2,p/3);


% External environment
mi_addblocklabel(s+10,p/3);

% Chamber wall material
mi_addblocklabel((b+a)/2,p/3);

% Inner magnet spacing material (starting from anode end)
mi_addblocklabel((b+c)/2,(g+f)/2);
mi_addblocklabel((b+c)/2,(k+h)/2);

% Outer sheilding (Fe)
mi_addblocklabel((e+r)/2,p/3);

%% Apply materials

% Magnets (starting from anode)
mi_selectlabel((b+d)/2,f/2);   
mi_setblockprop('SmCo 27 MGOe', 0, 1, '<None>', 90, 0, 0);
mi_clearselected

mi_selectlabel((b+d)/2,(g+h)/2);   
mi_setblockprop('SmCo 27 MGOe', 0, 1, '<None>', 270, 0, 0);
mi_clearselected

mi_selectlabel((b+d)/2,(k+l)/2);   
mi_setblockprop('SmCo 27 MGOe', 0, 1, '<None>', 90, 0, 0);
mi_clearselected

% Chamber wall 
mi_selectlabel((b+a)/2,p/3);   
mi_setblockprop('BN Ceramic', 0, 1, '<None>', 0, 0, 0);
mi_clearselected

% Magnetic guides / Inner Magnet spacers
mi_selectlabel((b+c)/2,(g+f)/2);   
mi_setblockprop('Pure Iron', 0, 1, '<None>', 0, 0, 0);
mi_clearselected

mi_selectlabel((b+c)/2,(k+h)/2);   
mi_setblockprop('Pure Iron', 0, 1, '<None>', 0, 0, 0);
mi_clearselected

% % % B/c FEMM get confused when running multiple instances and needs to
% % % allocate which window it has to concentrate on..
% % % HandleToFEMM = hand{id};

% Thruster housing
mi_selectlabel((d+e)/2,p/3);   
mi_setblockprop('Aluminum, 6061-T6', 0, 1, '<None>', 0, 0, 0);
mi_clearselected

mi_selectlabel((r+s)/2,p/3);   
mi_setblockprop('Aluminum, 6061-T6', 0, 1, '<None>', 0, 0, 0);
mi_clearselected

% Outer sheilding
mi_selectlabel((e+r)/2,p/3);   
mi_setblockprop('Pure Iron', 0, 1, '<None>', 0, 0, 0);
mi_clearselected

% External Enovironment
mi_selectlabel(s+10,p/3);   
mi_setblockprop('Air', 0, 1, '<None>', 0, 0, 0);
mi_clearselected

%% Analysis and display

mi_zoomnatural % zoom GUI view to see whole model

try
    mkdir([homefolder '/FEMM Analysis Files']);
catch
end
mi_saveas([pwd '/FEMM Analysis Files/CFT_Gen_' num2str(gen) '_id_' num2str(id) '.fem']); % Save current generation's model
% err = 0;
% err  = mi_analyze; %==================================================================================================================================THIS LINE
% if err == -1;
%     err = -1;
%     return
% else
mi_analyze
mi_loadsolution

%% Points of interest Data extraction

mo_seteditmode('contour')


% Flux magnitude @ thruster centreline
mo_addcontour(0.001,0); % Create contour
mo_addcontour(0.001,25);
try
    mkdir([homefolder '/FEMM data']);
catch
end
mo_makeplot(1,200,[pwd '/FEMM data/Flux Magnitude Channel Centreline_Gen_' num2str(gen) '_id_' num2str(id) '.txt'],0); % Save Data points to .txt file
mo_clearcontour

% Flux magnitude @ thrust chamber wall
mo_addcontour(2,0); % Create contour
mo_addcontour(2,25);
mo_makeplot(1,200,[pwd '/FEMM data/Flux Magnitude Channel Wall_Gen_' num2str(gen) '_id_' num2str(id) '.txt'],0); % Save Data points to .txt file
mo_clearcontour

%% Close FEMM
% end
closefemm

end



