function [label] = label_segment(C,ind)
%UNTITLED Summary of this function goes here

label = 0;

% Time boundaries
min_time = ind(1)/25;
max_time = ind(end)/25;

head=(C.Properties.VariableNames);
[~,sec_ind]=sort(head);

action_times = C(:,sec_ind(3)).Variables; %takes the correct col data
labels = C.Label;

for i=1:length(action_times)

    if action_times(i)>= min_time && action_times(i)<= max_time
        label = labels(i);

    end
end

end