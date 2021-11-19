function [features] = Window_features(acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z)
%  

features = zeros(1,48)-99;

% Acc_x features
[features(1),features(2)] = max(acc_x);
[features(3),features(4)] = min(acc_x);
features(5) = std(acc_x);
features(6) = median(abs(acc_x));
features(7) = bandpower(acc_x,25,[0 12.5]);
features(8) = iqr(timeseries(acc_x));

% Acc_y features
[features(9),features(10)] = max(acc_y);
[features(11),features(12)] = min(acc_y);
features(13) = std(acc_y);
features(14) = median(abs(acc_y));
features(15) = bandpower(acc_y,25,[0 12.5]);
features(16) = iqr(timeseries(acc_y));

% Acc_z features
[features(17),features(18)] = max(acc_z);
[features(19),features(20)] = min(acc_z);
features(21) = std(acc_z);
features(22) = median(abs(acc_z));
features(23) = bandpower(acc_z,25,[0 12.5]);
features(24) = iqr(timeseries(acc_z));

% Gyro x features
[features(25),features(26)] = max(gyro_x);
[features(27),features(28)] = min(gyro_x);
features(29) = std(gyro_x);
features(30) = median(abs(gyro_x));
features(31) = bandpower(gyro_x,25,[0 12.5]);
features(32) = iqr(timeseries(gyro_x));

% Gyro y features
[features(33),features(34)] = max(gyro_y);
[features(35),features(36)] = min(gyro_y);
features(37) = std(gyro_y);
features(38) = median(abs(gyro_y));
features(39) = bandpower(gyro_y,25,[0 12.5]);
features(40) = iqr(timeseries(gyro_y));

% Gyro z features
[features(41),features(42)] = max(gyro_z);
[features(43),features(44)] = min(gyro_z);
features(45) = std(gyro_z);
features(46) = median(abs(gyro_z));
features(47) = bandpower(gyro_z,25,[0 12.5]);
features(48) = iqr(timeseries(gyro_z));

end