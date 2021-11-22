function [label] = label_segment(C,ind)
% Label window according to labels file

label = 0;

% Time boundaries
min_time = ind(1)/25;
max_time = ind(end)/25;

% Sort column by names to overcome different column names
head=(C.Properties.VariableNames);
[~,sec_ind]=sort(head);

% Use the correct column, containing action times
action_times = C(:,sec_ind(3)).Variables; 
labels = C.Label;

% Check if window contains label
for i=1:length(action_times)

    if action_times(i)>= min_time && action_times(i)<= max_time
        label = labels(i);

    end
end

end