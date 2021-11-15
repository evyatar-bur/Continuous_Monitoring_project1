clc
clear
close all

Path = 'Good Recordings/8.9.Acc.csv';

% Read data from file
[t,x,y,z] = read_data(Path);
signals = {x,y,z};

% Define sampling frequency
if contains(Path,'Gyro')
    fs = 100; % Hz   
else
    fs = 25; % Hz
end


window_size = 100;

feature_num = 3*2*5;
window_num = length(x) - window_size + 1;


feature_values = zeros(window_num,feature_num);

for i= 1:window_num
    for axis = 1:3

        curr_signal = signals{axis};

        window = curr_signal(i:i+window_size-1);

        feature_values(i,(axis-1)*8+1:axis*8) = Window_features(window);
    end
end
