clc 
clear

window_size = 16;   % Sec
cut_flag = true;

true_cell = cell(2,1000);
check_cell = cell(2,1000);

max_last_window=ones(1,6); % for first window features calc

% Suppress readtable warning
warning('off','MATLAB:table:ModifiedAndSavedVarnames')

%% Section 1.a : Iterate to load files, extract features, and build matrix
sample_rate = 25;      

d=dir('*.Acc.csv');
X_event=zeros(50000,72)-99;    % Allocate memory for matrix X, with default value -99
Y_event=zeros(50000,1)-99;     % Allocate memory for label vector Y

n_instance=0;                  % Window counter

% make a High Pass Filter
fco = 0.1;                     % cutoff frequency (Hz)
Np = 2;                        % filter order=number of poles

[b,a]=butter(Np,fco/(sample_rate/2),'high'); 

% Iterate through recordings
for r=1:length(d)

    % Remember index for train/test partition
    if contains(d(r).name, '25') && cut_flag 

        cut_flag = false;
        cut_ind = n_instance;
    end
    
    % Read data from recordings
    A=readtable(d(r).name);
    gyro_file=strrep(d(r).name,'Acc','Gyro');
    B=readtable(gyro_file);
    label_file=strrep(d(r).name,'Acc','Label');
    C=readtable(label_file);
    acc_x=A.x_axis_g_;
    acc_y=A.y_axis_g_;
    acc_z=A.z_axis_g_;
    gyro_x=B.x_axis_deg_s_;
    gyro_y=B.y_axis_deg_s_;
    gyro_z=B.z_axis_deg_s_;

    % apply the filter only on acc recordings
    acc_x = filtfilt(b,a,acc_x); 
    acc_y = filtfilt(b,a,acc_y);
    acc_z = filtfilt(b,a,acc_z);

    % Check the minimum Length from the sensor
    N=length(acc_x);
    if length(gyro_x)<length(acc_x)
        N=length(gyro_x);
    end

    % Ignore recordings with significant difference between signal lengths
    if abs(length(gyro_x)-length(acc_x))>500
        disp(['Signal ignored - difference between signals is too large - ' d(r).name])
        continue
    end
    
    % Find suspected events for window
    [~,locs] = findpeaks(gyro_x,'MinPeakHeight',15,'MinPeakDistance',250);  
    
    % Suspected times vector, for accuracy check 
    suspected_times = [];
    
    % Iterate through suspected indices
    for i= 1:length(locs)
        
        % Window indexes
        min_ind = locs(i)-((window_size/2)*25);
        max_ind = locs(i)+((window_size/2)*25);
        
        ind = min_ind:max_ind;
        
        % Check if window exceeds record length
        if min_ind<1
            ind = 1:window_size*25;
        elseif max_ind>N
            ind = N-window_size*25:N;
        end
        
        % If window values reach treshold, compute features
        if std(acc_x(ind)) > 0.05    
            
            % Save times of suspected windows
            suspected_times(end+1) = locs(i)/25; 
            
            % Compute max of the previous window, for feature calculation
            last_window_ind = ind-window_size*25;

            if last_window_ind(1) > 0
                max_last_window(1) = max(acc_x(last_window_ind));
                max_last_window(2) = max(acc_y(last_window_ind));
                max_last_window(3) = max(acc_z(last_window_ind));
                max_last_window(4) = max(gyro_x(last_window_ind));
                max_last_window(5) = max(gyro_y(last_window_ind));
                max_last_window(6) = max(gyro_z(last_window_ind));
            end
            
            % Compute feature vector
            X_row = extract_features_204613681_308317361(acc_x(ind),acc_y(ind),acc_z(ind),gyro_x(ind),gyro_y(ind),gyro_z(ind),max_last_window);
            
            % Counter
            n_instance=n_instance+1;
            
            % Add feature vector to feature matrix
            X_event(n_instance,:) = X_row;
            Y_event(n_instance) = label_segment(C,ind,N);

        end
    end

    true_cell{1,r} = d(r).name;
    true_cell{2,r} = event_times(C);

    check_cell{1,r} = d(r).name;
    check_cell{2,r} = suspected_times;

end

% Delete empty cells
true_cell(:,r+1:end) = [];
check_cell(:,r+1:end) = [];

% Delete empty rows
ind = find(Y_event~=-99);
X_event = X_event(ind,:);
Y_event = Y_event(ind,:);

% Check event detection
[false_negetive,true_positive,bad_recordings] = check_event_detection(true_cell,check_cell);


%% Section 1.b. Features normalization
X_norm = normalize(X_event,1,'medianiqr');

disp('------------------------------------------')
disp('Features are after pre-processing! ')
disp('------------------------------------------')
% End Section 1.b.

%% Section 1.c. set training & Test sets

% Devide data to test and train - 8 last records are test data

% update the below sets
X_train=X_norm(1:cut_ind,:);
X_test=X_norm(cut_ind+1:end,:);
Y_train=Y_event(1:cut_ind);
Y_test=Y_event(cut_ind+1:end);

% End Section 1.c.

%% Section 1.d. remove correlated features

feature_names = {'max acc x','zero cross acc x','min acc x','diff acc x','std acc x','median acc x','bandpower acc x','mean squared acc x','skewness a x','max/last window acc x','tresh 25% acc x','tresh const acc x',...
    'max acc y','zero cross acc y','min acc y','diff acc y','std acc y','median acc y','bandpower acc y','mean squared acc y','skewness a y','max/last window acc y','tresh 25% acc y','tresh const acc y',...
    'max acc z','zero cross acc z','min acc z','diff acc z','std acc z','median acc z','bandpower acc z','mean squared acc z','skewness a z','max/last window acc z','tresh 25% acc z','tresh const acc z',...
    'max gyro x','zero cross gyro x','min gyro x','diff gyro x','std gyro x','median gyro x','bandpower gyro x','mean squared gyro x','skewness g x','max/last window gyro x','tresh 25% gyro x','tresh const gyro x',...
    'max gyro y','zero cross gyro y','min gyro y','diff gyro y','std gyro y','median gyro y','bandpower gyro y','mean squared gyro y','skewness g y','max/last window gyro y','tresh 25% gyro y','tresh const gyro y',...
    'max gyro z','zero cross gyro z','min gyro z','diff gyro z','std gyro z','median gyro z','bandpower gyro z','mean squared gyro z','skewness g z','max/last window gyro z','tresh 25% gyro z','tresh const gyro z'};

% Calculate feature weights with relief algorithm
Y_train_hat = (Y_train ~= 0);

len = size(X_train,2);
W = zeros(len,1);
for j=1:len
    [~,W(j)] = relieff(X_train(:,j),Y_train_hat,10);
end

% Sort features by feature weights
[~,ind] = sort(W,'descend');

X_train = X_train(:,ind);
X_test = X_test(:,ind);
X_norm = X_norm(:,ind);
feature_names = feature_names(ind);
W = W(ind);

% Delete corralated features
for i= 1:size(X_train,2)

    if i<size(X_train,2)

        R = corr(X_train,'type','Spearman');
        R = R(i,:);

        ind = (abs(R)>0.7 & R~=1);

        X_train(:,ind) = [];
        X_test(:,ind) = [];
        X_norm(:,ind) = [];
        feature_names(ind) = [];
        W(ind) = [];
    end
end

% End Section 1.d.

%% Section 2.a. Select the best feature

best_feature_list = [];
best_score = 0;
method = 'F1'; % 'F1', 'ROC' or 'PRC'

[best_feature_list,best_score] = Add_feature(X_train,X_test,Y_train,Y_test,best_feature_list,best_score,method);

% best feature
disp(['The best feature is number: ',num2str(best_feature_list(1)),' - ',feature_names{best_feature_list(1)}])
disp(['The best F1 score is: ',num2str(best_score)])
disp('------------------------------------------')


%% Section 2.b. Select the next best feature that is best together with the first feature

[best_feature_list,best_score] = Add_feature(X_train,X_test,Y_train,Y_test,best_feature_list,best_score,method);

% Print second best feature
disp(['The second best feature is number: ',num2str(best_feature_list(end)),' - ',feature_names{best_feature_list(end)}])
disp(['The best F1 score is: ',num2str(best_score)])
disp('------------------------------------------')
% End Section 2.b.

%% Add more features

for i = 1:8
    [best_feature_list,best_score] = Add_feature(X_train,X_test,Y_train,Y_test,best_feature_list,best_score,method);
    disp(['The new best feature is number: ',num2str(best_feature_list(end)),' - ',feature_names{best_feature_list(end)}])
    disp(['The best F1 score is: ',num2str(best_score)])
    disp('------------------------------------------')
end

%% Section 4 Train ensemble models

% Use only best features
train_data = X_train(:,best_feature_list);
test_data = X_test(:,best_feature_list);

% Tree settings
t = templateTree('MaxNumSplits',50,'Surrogate','on','SplitCriterion','deviance');
learning_rate = 0.01;

% Train RUSboost model on best features
Ensemble_bagging_MDL = fitcensemble(train_data,Y_train,'method','RUSBoost','NumLearningCycles',1000,'Learners',t,'LearnRate',learning_rate);

% End Section 4.

%% Section 5 display confusion matrix on test set

% Predict scores and predictions
[prediction,scores] = predict(Ensemble_bagging_MDL,test_data);

% Create confusion matrix
confusion_mat = confusionmat(Y_test,prediction);

% Combine horizontal and vertical zoom 
confusion_mat(8,:) = confusion_mat(8,:) + confusion_mat(9,:);
confusion_mat(:,8) = confusion_mat(:,8) + confusion_mat(:,9);
confusion_mat(:,9) = [];
confusion_mat(9,:) = [];

confusion_mat(6,:) = confusion_mat(6,:) + confusion_mat(7,:);
confusion_mat(:,6) = confusion_mat(:,6) + confusion_mat(:,7);
confusion_mat(:,7) = [];
confusion_mat(7,:) = []

figure()
xvalues={'no event', 'scroll up', 'scroll down', 'on/off','noise', 'zoom in','zoom out'};
confusionchart(confusion_mat,xvalues)

disp('------------------------------------------')
% End Section 5.

%% Section 6 Create final model for submission
Final_data = X_norm(:,best_feature_list);

Ensemble_bagging_MDL_4submission = fitcensemble(train_data,Y_train,'method','RUSBoost','NumLearningCycles',1000,'Learners',t,'LearnRate',learning_rate);

disp('------------------------------------------')

save('Event_trigger_model.mat','Ensemble_bagging_MDL_4submission')
% End Section 6.

