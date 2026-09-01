%
%   plot the Pareto optimal front and other solutions
%

clear all;
close all;

currentfolder = pwd;
% addpath('opt_utils'); % shouldn't this already be added?

set(0,'defaultAxesFontSize',14); set(0,'defaultAxesFontName','Times New Roman');
set(0,'defaultTextFontSize',14); set(0,'defaultTextFontName','Times New Roman');

%   set data information and other parameters (INPUT)
% path = '../../Projects/Optimisation/Injection/Injection_18th_AIAA_SPH/NEW';
% dfnCFD = [path '/f_Injection_para-SAEA_rev-CFD_Overall_rev.csv'];
% show   = struct('nondomin',1, 'baseline',1, 'selected',1);
% path = pwd;
% dfnCFT = [path '/f_Injection_Sur_para-SAEA_rev-CFD_plusD.csv'];
dfnCFT = [currentfolder '/CFTOpt-SAEA_rev-ND.dat'];
show   = struct('nondomin',1, 'baseline',1, 'selected',0);
addpath(currentfolder);
genmin = 0;
genmax = 50;
objcol = 4:6;               % column number(s) of objective  functions
concol = 8;                    % column number(s) of constraint functions
xiscol = 9:16;                 % column number(s) of decision variables
range  = [-80 0 -100 0 -4000 0]; %[-300 0 -100 0 -10000 0];
objsgn = [1 1 1];
lblstr = {'T [mN]','n_{t} [%]','Isp [s]','P [W]'}; % 
xis4cb = 7;             % column in data with 4th objective function        % decision variable used for colour dots (6:eta_m, 7:Dp0, 8:hf)
xisstr = {'P [W]'};
xrange = {[0.001 1000],[0.001 10],[0.01 50],[2 50],[2 50],[2 50],[2 50],[2 50]}; 
vangle = [-113 10];

%% Selected Individuals - select a few points from pareto front to single out

%   set values for selected individuals % g35i36, g04i08, g12i36, g02i19)
% xs = [-0.206129 -0.184267 -0.191948 -0.130421];   % mixing efficiency
% ys = [ 0.519840  0.461648  0.556999  0.655495];   % total pressure loss
% zs = [-0.753422 -0.654469 -0.864172 -0.687625];   % penetration height

% %   set values for selected individuals % g40i27, g06i38, g33i31, g02i06)
% xs = [-0.210242 -0.152588 -0.201034 -0.131818];   % mixing efficiency
% ys = [ 0.542373  0.444985  0.565396  0.670563];   % total pressure loss
% zs = [-0.796875 -0.607016 -0.876875 -0.693844];   % penetration height

%%   baseline values - define your baseline here - Baseline is from A. Keller's Phd thesis
% each camparitor point is represented in the column created when aligning
% / stacking the x,y,z,P vectors
xbl = [0.120 0.270 0.290].*(-1);   % Thrust [mN]
ybl = [0.377 0.407 0.437].*(-100);   % efficiency [%] = 0.407 +/- 0.03
zbl = [650 1090 1620].*(-1);   % Isp [s] <-- these are Isp,ion. Isp,tot = [220 555 745]
Pbl = [4.2 9.3 12.1];   % Power

%%   load data for CFD-evaluated individuals
data = loadOptValues(dfnCFT);

% [~,data] = loadVarsData(dfnCFT); % Commented this out // seems like a
% useless function really... why convert to .csv when it's already a dat
% file? --> angus

%data = importdata(dfnCFT);
gens = data(:,1);       % 1st column: generation ID
inds = data(:,2);       % 2nd column: individual ID
feas = data(:,3);       % 3rd column: feasibility
objs = data(:,objcol);  % objective  functions
cons = data(:,concol);  % constraint functions
xiss = data(:,xis4cb);  % get 4th objective function % xiss = data(:,xiscol); % decision variables
dlen = size(data,1);
dcol = size(data,2);
X = objs(:,1);          % Thrust [mN]
Y = objs(:,2);          % Efficiency [%]
Z = objs(:,3);          % Isp [s]

%   consider only data with the range of interest
if genmin <= 0, genmin = 1;         end
if genmax <= 0, genmax = max(gens); end
feasibles = find( gens >= genmin & gens <= genmax & feas == 1);                 %   feasible solutions

%   obtain solutions on Pareto optimum front
pop.nf   = length(objcol);
pop.f    = objs;
pop.g    = cons;
pop.feas = feas;
pop.size = dlen;
for i = 1:dlen, pop = assign_fitness(pop, i, pop.f(i,:), pop.g(i,:)); end
pop = sort_nd_maxcv(pop);
optimumms = pop.nd_rank == 1;
%optimumms = pop.rank(1:N);

%% Plot Options

%   prepare colormap for grayscale (not using greyscale)
cmap_jet = jet(256);
% [~,idx]  = sortrows(rgb2hsv(cmap_jet), -1); % sort by hue
% cmap_gray = gray(256)*0.85+0.15;            % mitigate darkness
% cmap_gray = cmap_gray(idx,:);

%   set colour bar
h = figure('NextPlot', 'add'); figure(h);
cmin = min(0);   % Minimum for 4th objeective function: Power                          % xrange{xis4cb});
cmax = max(2000);  % Minimum for 4th objeective function: Power                       % xrange{xis4cb});
caxis([cmin cmax]);
colorbar;
colorbar('location','EastOutside');
narrowColorBars; 
%cmap= colormap('jet');
cmap = colormap(cmap_jet);
cblabel(['\it' xisstr]);  % {xis4cb}

%   plot solutions in 3D space
hold on;
grid on;
for i = 1:dlen;
    if ~optimumms(i) || ~show.nondomin
%         if xis4cb <=5,
            val = xiss(i,1);
%         else
%             val = objs(i,xis4cb-5) * objsgn(xis4cb-5);
%         end
        c = cmap(fix(max(min(val,cmax)-cmin,0)/(cmax-cmin)*(length(cmap)-1))+1,:);
        h1 = plot3(X(i), Y(i), Z(i), 'ko', 'MarkerSize',5, 'MarkerFaceColor',c);
    end
end
for i = 1:dlen;
    if optimumms(i) && show.nondomin
%         if xis4cb <=5, 
            val = xiss(i,1);
%         else
%             val = objs(i,xis4cb-5) * objsgn(xis4cb-5);
%         end
        c = cmap(fix(max(min(val,cmax)-cmin,0)/(cmax-cmin)*(length(cmap)-1))+1,:);
        h2 = plot3(X(i), Y(i), Z(i), 'kd', 'MarkerSize',6, 'MarkerFaceColor',c);
    end
end
if show.selected, hs = plot3( xs , ys , zs , 'ro', 'MarkerSize',8);                               end
if show.baseline, h0 = plot3( xbl, ybl, zbl, 'ks', 'MarkerSize',6, 'MarkerFaceColor',[.6 .6 .6]); end
if show.nondomin, appendLegend(h2, 'non-dominated', 1); end
if show.baseline, appendLegend(h0, 'baseline'     , 1); end
if show.selected, appendLegend(hs, 'selected'     , 1); end
h = legend;
xlabel(['\it' lblstr{1}]);
ylabel(['\it' lblstr{2}]);
zlabel(['\it' lblstr{3}]);
axis(range);
axis square;
set(gca,'YDir','reverse');
camproj('orthographic');
hold off;

%   various views
%view(vangle);                                  % 3D view
%view(-90,90); set(h,'Location','SouthEast');   % (b) Delta p_0 vs -eta_m
view(  0,  0); set(h,'Location','SouthEast');  % (c)    -eta_m vs -h_p
%view(-90,  0); set(h,'Location','SouthEast');  % (d) Delta p_0 vs -h_p