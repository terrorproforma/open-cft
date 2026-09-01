%
%   build and save surrogate models from exisiting data for truly-evaluated solutions
%

% clear all;

%   set data information and other parameters (INPUT)
path= '.';
opt2 = '.'; %Replace with path to opt2
addpath(path,opt2,'-begin');
dfnEVS = [pwd '/CFTOpt-SAEA_rev-EVS.dat']; %Replace OptName
dfnPrm = [pwd '/params.m'];
dfnSur = regexprep(regexprep(regexprep(dfnEVS,'\EVS','Sur'),'\csv','dat'),'\.dat','.mat');
dfnRng = regexprep(regexprep(regexprep(dfnEVS,'\EVS','Rng'),'\csv','dat'),'\.dat','.mat');
newRng = 1; % 0: load rng 1: use and save current rng
genmin = 1;
genmax = 50;
objiSA = 1:4;           % objective functions for sensitivity analysis
objcol = 4:7;           % column number(s) of objective functions
xiscol = 9:16;          % column number(s) of decision variables

%   load data for Truely-evaluated individuals
data = loadOptValues(dfnEVS);
gens = data(:,1);       % 1st column: generation ID
inds = data(:,2);       % 2nd column: individual ID
feas = data(:,3);       % 3rd column: feasibility
objs = data(:,objcol);  % objective functions
xiss = data(:,xiscol);  % decision variables
dlen = size(data,1);
dcol = size(data,2);

%   consider only data with the range of interest
if genmin <= 0, genmin = 1;         end
if genmax <= 0, genmax = max(gens); end
% valid = find( gens >= genmin & gens <= genmax );    % consider all evaluated solutions
valid = find( gens >= genmin & gens <= genmax & feas == 1); % only feasible solutions

%   prepare param file in 'scripts' directory
%copyfile( dfnPrm, 'params.m' );

%   save random number seed
if newRng
    rngstr = rng; %#ok<*UNRCH>
    save(dfnRng, 'rngstr');
    disp(['Random No seed ''rngstr'' written into file ''' dfnRng '''.']);
else
    load(dfnRng);
    rng(rngstr);
    disp(['Random No seed ''rngstr''  loaded from file ''' dfnRng '''.']);
end    

%   build surrogate models
sptstr = strsplit_rev(dfnEVS,'-SAEA'); sptstr = strsplit_rev(sptstr{1},'/');
fnSAEA = sptstr{end};   % name of SAEA function (ex. f6vSCR_para)
f = feval(fnSAEA);
surr = Surrogate();
surr = set_range(surr, f);
surr = add_points(surr, xiss(valid,:), objs(valid,objiSA));
surr = train_rev(surr);
save(dfnSur, 'surr');
disp(['Surrogate models ''surr'' written into file ''' dfnSur '''.']);

%   delete param file in 'scripts' directory
%delete( 'params.m' );