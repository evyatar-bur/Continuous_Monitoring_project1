function [label] = label_segment(C,ind,N)
% Label window according to labels file

label = 0;

% Time boundaries
min_time = ind(1)/25 -(ind(1)/N)*10 ;
max_time = ind(end)/25 -(ind(1)/N)*3 ;

% Use the correct column, containing action times
action_times = event_times(C);
labels = C.Label;

% Check if window contains label
for i=1:length(action_times)

    if action_times(i) >= min_time && action_times(i) <= max_time
        label = labels(i);

    end
end

end