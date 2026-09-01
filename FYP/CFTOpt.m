function [f, g] = CFTOpt(x,gen,id,mse)
if nargin == 0
    prob.nx = 8; %  Number of descicion variables - inputs to code
    prob.nf = 4;  %  Number of objective functions - 
    prob.ng = 1;  %  Number of constraint functions

    prob.range = cell(prob.nx, 1);
    % POWER
    prob.range{1} = Range('range', [1 1000]); % Voltage Range
    prob.range{2} = Range('range', [0.001 10]); % Current Range
    % MASS FLOW RATE
    prob.range{3} = Range('range', [0.2 50]);
    % GEOMETRY
    prob.range{4} = Range('range', [2 50]);% IMR
    prob.range{5} = Range('range', [2 50]);% OMR
    prob.range{6} = Range('range', [2 50]);% inner sheild radius
    % iron sheild
    prob.range{7} = Range('range', [2 50]); % outer sheild radius
    prob.range{8} = Range('range', [2 50]); %outer shell radius
   
    
    f = prob;
else
    [f, g] = CFTScript_true(x,gen,id,mse); 
end
return

function [f, g] = CFTScript_true(x,gen,id,~) 
    [output, g] = Performance_est(x,gen,id);
    f(1) = output.f1;
    f(2) = output.f2;
    f(3) = output.f3;
    f(4) = output.f4;

disp(['< Gen = ' num2str(gen) ' , Ind = ' num2str(id) ' >']);
disp(['Thrust = ' num2str(f(1)) ' mN']);
disp(['Total Efficiency = ' num2str(f(2)) ' %']);
disp(['Isp = ' num2str(f(3)) ' s']);
disp(['Power = ' num2str(f(4)) ' W']);
disp(' ');
return