function [best_feature_list,best_score] = Add_feature(X_training,X_test,Y_training,Y_test,best_feature_list,best_score,metric)
% Add_features checks which feature needs to be added to get the best AUC

prev_best_score = best_score;
best_score = 0;

% Iterate through features to find best one to add 
for i = 1:size(X_training,2)

    if ismember(i,best_feature_list)
        continue
    end
    
    % Use best features, along with feature i
    train_data = X_training(:,[best_feature_list i]);
    test_data = X_test(:,[best_feature_list i]);
    
    % Train model
    model=fitensemble(train_data,Y_training,'Bag',100,'Tree','Type','classification');
    
    % Predict scores
    [prediction,scores] = predict(model,test_data);
    
    if strcmp(metric,'ROC')
        % Compute AUC - ROC
        [~,~,~,score] = perfcurve(Y_test,scores(:,1),'0');

    elseif strcmp(metric,'PRC')
        % Compute AUC - PRC
        [~,~,~,score] = perfcurve(Y_test,scores(:,1),'0','XCrit','tpr','YCrit','ppv');

    elseif strcmp(metric,'F1')
        % Compute f1 score
        tp = sum((prediction ~= 0) & (Y_test ~= 0));
        fp = sum((prediction ~= 0) & (Y_test == 0));
        fn = sum((prediction == 0) & (Y_test ~= 0));

        precision = tp / (tp + fp);
        recall = tp / (tp + fn);
        score = (2 * precision * recall) / (precision + recall);


    else
        disp("Unknown metric - choose 'ROC','F1 or 'PRC' ")

    end

    % Save index of the feature that improves AUC the best
    if score>best_score
        
        best_score = score;
        best_feature_ind = i;

    end
end

% Check if AUC improved significantly, if so - add best feature
%if best_AUC-prev_best_AUC>0.0001
    best_feature_list(end+1) = best_feature_ind;
%else
%    disp('Best AUC not improved - Do not add more features')

end