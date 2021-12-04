function [false_negetive,true_positive,bad_recordings] = check_event_detection(true_cell,check_cell)

false_negetive = 0;
event_count = 0;

% Cell of the recording containing misses
bad_recordings = {};

for i = 1:size(true_cell,2)

    suspected_times = check_cell{2,i};
    true_time = true_cell{2,i};

    for j = 1:length(true_time)
        
        % Count events and check if each event was spotted
        event_count = event_count + 1;
        diff_vec = abs(suspected_times-true_time(j));
        
        if sum(diff_vec<16) == 0
            false_negetive = false_negetive + 1;

            bad_recordings{1,end+1} = true_cell{1,i}; 
            bad_recordings{2,end} = true_time; 
        end
    end
end

true_positive = event_count - false_negetive;

end