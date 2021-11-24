clc
% clear
close all

Path = '4.6.Acc.csv';

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
% plot(t,z)c xv c v cvcv xc xv

title('Measurments as a function of time')
xlabel('Time [sec]')
ylabel(y_label)
legend('x')

% max_val = max([max(x),max(y),max(z)]);
% 
% x_freq=fft(x(430:450));
% x2_freq=fft(x(200:2202));
% L = length(x(430:450));

% figure(1)
% 
% subplot(3,1,1)
% P2 = abs(x_freq/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% 
% f = 25*(0:(L/2))/L;
% plot(f,P1) 
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
% 
% subplot(3,1,2)
% P2 = abs(x2_freq/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% f = 25*(0:(L/2))/L;
% plot(f,P1) 
% title('Single-Sided Amplitude Spectrum of Y(t)')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
% 


% make a filter and apply it to the signal
fco=1;          %cutoff frequency (Hz)
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

