function [features] = Window_features(window)
%  

features = zeros(1,8);

[features(1),features(2)] = max(window);
[features(3),features(4)] = min(window);
features(5) = std(window);
features(6) = median(abs(window));
features(7) = zerocrossrate(window);
features(8) = iqr(timeseries(window));

end