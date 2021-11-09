clc
clear
close all

Path = 'Good Recordings/8.9.Acc.csv';

% Read data from file
[t,x,y,z] = read_data(Path);

% Define sampling frequency
if contains(Path,'Gyro')
    fs = 25; % Hz   
    y_label = 'Angular acceleration [deg/sec]';
else
    fs = 25; % Hz
    y_label = 'Acceleration [g]';
end
    
% Plotting signals
figure(1)
hold on

plot(t,x)
% plot(t,y)
% plot(t,z)c xv c v cvcv xc xv

title('Measurments as a function of time')
xlabel('Time [sec]')
ylabel(y_label)
legend('x','y','z')

max_val = max([max(x),max(y),max(z)]);




