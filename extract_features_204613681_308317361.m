function features=extract_features_204613681_308317361(acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z)
% This function recieves a window with six signals and returns the window's features

feature_num = 9;

features = zeros(1,feature_num*6)-99;
signals = {acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z};

for i = 1:6
    
    % Assign current signal
    signal = signals{i};
    
    % Set Starting index for current signal
    start_ind = (i-1)*feature_num+1;

    % Compute and assign signal features
    features(start_ind) = max(signal);
    features(start_ind+1) = Our_zero_crossing(signal);
    features(start_ind+2) = min(signal);
    features(start_ind+3) = sum(abs(diff(signal)));
    features(start_ind+4) = std(signal);
    features(start_ind+5) = median(abs(signal));
    features(start_ind+6) = bandpower(signal,25,[0 12.5]);
    features(start_ind+7) = iqr(timeseries(signal));
    features(start_ind+8) = skewness(signal);

end

end