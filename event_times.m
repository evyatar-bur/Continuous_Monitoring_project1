function [action_times] = event_times(C)

% Sort column by names to overcome different column names
head = (C.Properties.VariableNames);
[~,sec_ind]=sort(head);

% Use the correct column, containing action times
action_times = C(:,sec_ind(3)).Variables;

end