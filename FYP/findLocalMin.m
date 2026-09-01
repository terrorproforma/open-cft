%% Find Local Minimums - A.Muffatti 10/5/2016


function LocalMin = findLocalMin(Bdata,Ldata)

% data(:,1) = B_data.(B_dataname{i})(:,2);
% data(:,2) = B_data.(B_dataname{i})(:,1);

% Bdata = data(:,1);
% Ldata = data(:,2);

Bdata = (Bdata.*Bdata).^(1/2);

locpos = 1;
leng = length(Bdata);
for i = 2:leng-1 %the length of the data
    % if the current location is smaller than its neighbours then save its
    % location as a minimum
    if Bdata(i-1) > Bdata(i) && Bdata(i) < Bdata(i+1);
        locate = find(Bdata==Bdata(i));
        min(locpos,1) = Bdata(locate);
        min(locpos,2) = Ldata(locate);
        locpos = locpos + 1;
    end
    %find the locations of the minimums in the vector
    
end

LocalMin = min;

end