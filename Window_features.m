function [features] = Window_features(window, feature_num)
%  

features = zeros(1,feature_num);

[features(1),features(2)] = max(window);
[features(3),features(4)] = min(window);
features(5) = std(window);
features(6) = median(abs(window));
features(7) = zerocrossrate(window);
% features(6) = iqr(window');

end