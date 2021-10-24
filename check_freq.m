clc
clear

file_name = 'Gyroscope_6.csv';

% Read datafrom file
[t,x,y,z] = read_data(file_name);

% Define sampling frequency
if contains(file_name,'Gyro')
    fs = 100; % Hz   
else
    fs = 25; % Hz     
end

L = length(x);


x_freq = fft(x) ;

y_freq = fft(y) ;

z_freq = fft(z) ;

figure(1)

subplot(3,1,1)
P2 = abs(x_freq/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = fs*(0:(L/2))/L;
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

subplot(3,1,2)
P2 = abs(y_freq/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = fs*(0:(L/2))/L;
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of Y(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

subplot(3,1,3)
P2 = abs(z_freq/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = fs*(0:(L/2))/L;
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of Z(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

%%

new_x = x(1950:2200);


L = length(new_x);
new_x_freq = fft(new_x) ;

P2 = abs(new_x_freq/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = fs*(0:(L/2))/L;
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of Z(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

