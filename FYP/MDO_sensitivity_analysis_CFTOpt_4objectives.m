%% plot sensitivity analysis - Angus Muffatti 20/09/2016

clear
currentpath = pwd;

Sidata = importdata([currentpath '/CFTOpt-sensitivity_g050_Si_N10000.csv']); % get first order sensitivity
STidata = importdata([currentpath '/CFTOpt-sensitivity_g050_STi_N10000.csv']); % get global sensitivity

OFstring = {'T [mN]','n_{t}[%]','Isp [s]','P_{a} [W]'}; % Objective Functions
DVstring = {'\Phi_{a} [V]','I_{a} [A]','\deltam [sccm]','IMR','OMD','ISR','OSR','OER'}; % Decision Variables

[row,col] = size(Sidata); % [rows,columns]
SAdata = [];%zeros(row,2*col);
for i = 1:row;
    SAdata = [SAdata; Sidata(i,:); STidata(i,:)]; 
end

SAdata2 = abs([SAdata(1:4,:); SAdata(7:8,:)]); 

bar(abs(SAdata2),'stacked')
legend(DVstring)
% Create xlabel
xlabel('T [mN]                              n_{t}[%]                              P_{a} [W]  ');
% Set the remaining axes properties
set(gca,'XTick',[1 2 3 4 5 6],'XTickLabel',...
    {'Si','STi','Si','STi','Si','STi'});

fig = gcf;
figure(fig.Number+1)

bar(abs(SAdata),'stacked')
legend(DVstring)
% Create xlabel
xlabel('T [mN]                    n_{t}[%]                        Isp [s]                    P_{a} [W] ');
% Set the remaining axes properties
set(gca,'XTick',[1 2 3 4 5 6 7 8],'XTickLabel',...
    {'Si','STi','Si','STi','Si','STi','Si','STi'});