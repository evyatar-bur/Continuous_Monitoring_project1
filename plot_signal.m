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
    
% Plotting signals
figure(1)
hold on

plot(t,x)
plot(t,y)
plot(t,z)

title('Measurments as a function of time')
xlabel('Time [sec]')
ylabel(y_label)
legend('x','y','z')

max_val = max([max(x),max(y),max(z)]);


%% Cutting and labeling

% Plotting signals
figure(2)
hold on

plot(x)
% plot(y)
% plot(z)

forward_time_stamps = [140,180,260,300,400,440,510,550,640,680,];

