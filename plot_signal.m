clc
clear
close all

Path = '24.1.Gyro.csv';

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
% xlim([590 610])
% plot(t,y)
% plot(t,z)

title('Measurments as a function of time')
xlabel('Time [sec]')
ylabel(y_label)
legend('x')


% make a filter and apply it to the signal
fco=0.1;          %cutoff frequency (Hz)
Np=2;           %filter order=number of poles

[b,a]=butter(Np,fco/(fs/2),'high'); %high pass Butterworth filter coefficients
x_filt = filtfilt(b,a,x); %apply the filter to x(t)

% Plotting signals
figure(2)
hold on

plot(t,x_filt)

title('Measurments as a function of time')
xlabel('Time [sec]')
ylabel(y_label)
legend('x')

