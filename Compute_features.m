clc
clear
close all

Path = 'Good Recordings/8.5.Gyro.csv';

% Read data from file
[t,x,y,z] = read_data(Path);

% Define sampling frequency
if contains(Path,'Gyro')
    fs = 100; % Hz   
    y_label = 'Angular acceleration [deg/sec]';
else
    fs = 25; % Hz
    y_label = 'Acceleration [g]';
end


window_size = 100;
stride = 1;

feature_num = 3*2*5;
window_num = length(x) - window_size + 1;


feature_values = zeros(window_num,feature_num);

for i= 1:window_num

    window = x(i:i+window_size-1);

    feature_values(i,1:7) = Window_features(window, 7);

end
