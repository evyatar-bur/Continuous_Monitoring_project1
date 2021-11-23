clc
% clear
close all

Path = '12.6.Acc.csv';

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
legend('x','y','z')

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
