%% cusp_prob is a script for automatically reading the data from FEMM and - A.Muffatti 29/04/2016
% determining the cusp arrival probabilities to pass to the solver.

function [p1,p2,p3,p4] = cusp_prob(x,gen,id)

%% Insert routine here for Identifying the peak field strengths at each cusp
global homefolder;
dirdata = dir(fullfile([homefolder '/FEMM data'],['*_' num2str(gen) '_id_' num2str(id) '.txt']));

% Cusp locations - e = end, s = start

c4s_wall = 0;
c4e_wall = 0.5;
c3s_wall = 4.4;
c3e_wall = 5.0;
c2s_wall = 15;
c2e_wall = 15.6;
c1s_wall = 20;
c1e_wall = 21;
c4s_cl = 0;
c4e_cl = 0.5;
c3s_cl = 4.4;
c3e_cl = 5.0;
c2s_cl = 15;
c2e_cl = 15.6;
c1s_cl = 24;
c1e_cl = 25;

% % % Need to take positions from engine being analyzed - These are from
% % % CHengyu Ma et. al. [7]
% % % c4s_wall = 0;
% % % c4e_wall = 1;
% % % c3s_wall = 15;
% % % c3e_wall = 17;
% % % c2s_wall = 91;
% % % c2e_wall = 93;
% % % c1s_wall = 97;
% % % c1e_wall = 98;
% % % c4s_cl = 0;
% % % c4e_cl = 1;
% % % c3s_cl = 19.5;
% % % c3e_cl = 21.5;
% % % c2s_cl = 89.5;
% % % c2e_cl = 91.5;
% % % c1s_cl = 97;
% % % c1e_cl = 98;

for i=1:length(dirdata);
    namedata = {dirdata.name};
    filename = namedata{i};
    B_dataname{i} = matlab.lang.makeValidName(filename);
    B_data.(B_dataname{i}) = importfile([homefolder '/FEMM data/' filename]);
end

% fig = gcf; % get figure handle

% Flux Magnitude at channel wall must be handled First, Centreline second
% the

for i=1:length(B_dataname);
    if ~isempty(strfind(B_dataname{i},'Wall')); % 'Wall' - Looking for local maximums
        
%         figure(i);%fig.Number+i); set the figure number
        
        % get the magnitude of the data to determine maximums
        B_data.(B_dataname{i})(:,2) = (B_data.(B_dataname{i})(:,2).*B_data.(B_dataname{i})(:,2)).^(1/2);
        
%         plot(B_data.(B_dataname{i})(:,1),B_data.(B_dataname{i})(:,2)) %plot data
%         legename{i} = regexprep(B_dataname{i},'_txt',''); % save data names for legend
        
        % cusp data locations from [7] Chengyu Ma et al.
        %cusp 4
        Idx.cusp4 = not(abs(sign(sign(c4s_wall - B_data.(B_dataname{i})(:,1)) + sign(c4e_wall - B_data.(B_dataname{i})(:,1)))));
        %cusp 3
        Idx.cusp3 = not(abs(sign(sign(c3s_wall - B_data.(B_dataname{i})(:,1)) + sign(c3e_wall - B_data.(B_dataname{i})(:,1)))));
        %cusp 2
        Idx.cusp2 = not(abs(sign(sign(c2s_wall - B_data.(B_dataname{i})(:,1)) + sign(c2e_wall - B_data.(B_dataname{i})(:,1)))));
        %cusp 1
        Idx.cusp1 = not(abs(sign(sign(c1s_wall - B_data.(B_dataname{i})(:,1)) + sign(c1e_wall - B_data.(B_dataname{i})(:,1)))));
        
        % Location of data
        pos4 = find(Idx.cusp4==1);
        pos3 = find(Idx.cusp3==1);
        pos2 = find(Idx.cusp2==1);
        pos1 = find(Idx.cusp1==1);
        
        %     if ~isempty(strfind(B_dataname{i},'Wall')); % 'Wall' - Looking for local maximums
        for ix = 1:4; %number of cusps
            if ix == 1;
                loc = pos4;
            elseif ix == 2;
                loc = pos3;
            elseif ix == 3;
                loc = pos2;
            elseif ix == 4;
                loc = pos1;
            end
            for ii = 1:length(loc);
                Bdat(ii,1) = B_data.(B_dataname{i})(loc(ii),2);
            end
            Maxima(ix) = mean(Bdat);
            clear Bdat
        end
        
        % set graph properties
%         title(['\fontsize{16}' legename{i}]);
%         legend(legename{i});
%         xlabel('Length [mm]');
%         ylabel('Magnetic Field Strength [T]')
%         vline([c4s_wall c4e_wall c3s_wall c3e_wall c2s_wall c2e_wall c1s_wall c1e_wall],{'r','r','r','r','r','r','r'},{'   c4','','   c3','','  c2','','','         c1'});
        
    else
        
%         figure(i);%fig.Number+i); set the figure number
        
        % get the magnitude of the data to determine maximums
        B_data.(B_dataname{i})(:,2) = (B_data.(B_dataname{i})(:,2).*B_data.(B_dataname{i})(:,2)).^(1/2);
        
%         plot(B_data.(B_dataname{i})(:,1),B_data.(B_dataname{i})(:,2)) %plot data
        legename{i} = regexprep(B_dataname{i},'_txt',''); % save data names for legend
        
        % cusp data locations from [7] Chengyu Ma et al.
        %cusp 4
        Idx.cusp4 = not(abs(sign(sign(c4s_cl - B_data.(B_dataname{i})(:,1)) + sign(c4e_cl - B_data.(B_dataname{i})(:,1)))));
        %cusp 3
        Idx.cusp3 = not(abs(sign(sign(c3s_cl - B_data.(B_dataname{i})(:,1)) + sign(c3e_cl - B_data.(B_dataname{i})(:,1)))));
        %cusp 2
        Idx.cusp2 = not(abs(sign(sign(c2s_cl - B_data.(B_dataname{i})(:,1)) + sign(c2e_cl - B_data.(B_dataname{i})(:,1)))));
        %cusp 1
        Idx.cusp1 = not(abs(sign(sign(c1s_cl - B_data.(B_dataname{i})(:,1)) + sign(c1e_cl - B_data.(B_dataname{i})(:,1)))));
        
        % Location of data
        pos4 = find(Idx.cusp4==1);
        pos3 = find(Idx.cusp3==1);
        pos2 = find(Idx.cusp2==1);
        pos1 = find(Idx.cusp1==1);
        
        for ix = 1:4; %number of cusps
            if ix == 1;
                loc = pos4;
            elseif ix == 2;
                loc = pos3;
            elseif ix == 3;
                loc = pos2;
            elseif ix == 4;
                loc = pos1;
            end
            for ii = 1:length(loc);
                Bdat(ii,1) = B_data.(B_dataname{i})(loc(ii),2);
            end
            Minima(ix) = mean(Bdat);
            clear Bdat
        end
    
%          % set graph properties
%         title(['\fontsize{16}' legename{i}]);
%         legend(legename{i});
%         xlabel('Length [mm]');
%         ylabel('Magnetic Field Strength [T]')
%         vline([c4s_cl c4e_cl c3s_cl c3e_cl c2s_cl c2e_cl c1s_cl c1e_cl],{'r','r','r','r','r','r','r'},{'   c4','','   c3','','  c2','','','         c1'});
    end
end
% cd(homefolder)

% sort data extracted from files into low field and high field regions

% for each cusp determine the minimum magnitude of the B field (cenre of
% the magnet) and the magnitude of the highest field region (the cusp area)

%% Read the Maximum data points from the FEMM data.

% Low field - cusp region
B0 = Minima';

% High field - Cusp region
BM = Maxima'; %test value

% Re arrange b/c of anode condition - high field region is at the
% centreline and lowfield region is centreline of cusp 3
BM(1) = B0(1);
B0(1) = B0(2);

%% Calculate probability
p = zeros(4);
for i = 1:length(B0);
    
    theta_m = asin(sqrt(B0(i,1)/BM(i,1))); % acceptance angle
    func = @(theta) sin(theta);
    p(i) = integral(func,0,theta_m)*(2*pi)/(4*pi); % cusp arrival probability
    
end
p1 = p(4);
p2 = p(3);
p3 = p(2);
p4 = p(1);
end



