
% CHANGE FILE NAME!!!
% REPLACE ID NUMBER with YOUR ID
clc 
clear

window_size = 20;   % Sec
over_lap = 10;      % Sec

% Suppress readtable warning
warning('off','MATLAB:table:ModifiedAndSavedVarnames')

%% Section 1.a : Iterate to load files, extract features, and build matrix
sample_rate=25;      

d=dir('*.Acc.csv');
X=zeros(50000,48)-99;    % Allocate memory for matrix X, with default value -99
Y=zeros(50000,1)-99;     % Allocate memory for label vector Y

n_instance=0; % Window counter

for r=1:length(d)

    disp(d(r).name)

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

    % Moving window with over lap Predetermined
    n_segments=floor((N/sample_rate)/over_lap)-1;
    
    % Compute features for each window
    for segment=1:n_segments
        ind=(segment-1)*over_lap*sample_rate+(1:(sample_rate*window_size));
        X_row = extract_features_32132132(acc_x(ind),acc_y(ind),acc_z(ind),gyro_x(ind),gyro_y(ind),gyro_z(ind));
        
        n_instance=n_instance+1;

        X(n_instance,:) = X_row;
        Y(n_instance) = label_segment(C,ind);
    end
end

% Delete empty rows
ind = find(Y~=-99);
X = X(ind,:);
Y = Y(ind,:);

%% Section 1.b. Features normalization/discretization remove outliers if needed
X_norm = normalize(X,1,"medianiqr");
% update the above matrices after discretization
disp('------------------------------------------')
disp('Features are after pre-processing! ')
disp('------------------------------------------')
% End Section 1.b.

%% Section 1.c. set training & Test sets

% Devide data to test and train - 8 last records are test data

% update the below sets
X_train=X_norm(1:34029,:);
X_test=X_norm(34030:end,:);
Y_train=Y(1:34029);
Y_test=Y(34030:end);

% End Section 1.c.

%% Section 1.d. remove correlated features

feature_names = {'max_acc_x','max_ind_acc_x','min_acc_x','min_ind_acc_x','std_acc_x','median_acc_x','bandpower_acc_x','iqr_acc_x',...
    'max_acc_y','max_ind_acc_y','min_acc_y','min_ind_acc_y','std_acc_y','median_acc_y','bandpower_acc_y','iqr_acc_y',...
    'max_acc_z','max_ind_acc_z','min_acc_z','min_ind_acc_z','std_acc_z','median_acc_z','bandpower_acc_z','iqr_acc_z',...
    'max_gyro_x','max_ind_gyro_x','min_gyro_x','min_ind_gyro_x','std_gyro_x','median_gyro_x','bandpower_gyro_x','iqr_gyro_x',...
    'max_gyro_y','max_ind_gyro_y','min_gyro_y','min_ind_gyro_y','std_gyro_y','median_gyro_y','bandpower_gyro_y','iqr_gyro_y',...
    'max_gyro_z','max_ind_gyro_z','min_gyro_z','min_ind_gyro_z','std_gyro_z','median_gyro_z','bandpower_gyro_z','iqr_gyro_z'};


% Delete corralated features
for i= 1:size(X_train,2)

    if i<size(X_train,2)

        R = corr(X_train,'type','Spearman');
        R = R(i,:);

        ind = (R>0.7 & R~=1);

        X_train(:,ind) = [];
        X_test(:,ind) = [];
        feature_names(ind) = [];
    end
end

% End Section 1.d.


%% Section 2.a. Select the best feature

best_feature_list = [];
best_AUC = 0;
method = 'PRC';

[best_feature_list,best_AUC] = Add_feature(X_train,X_test,Y_train,Y_test,best_feature_list,best_AUC,'PRC');

% update the above parameter based on a criterion you choose to select the
% best feature
disp(['The best feature is number: ',num2str(best_feature_list(1)),' - ',feature_names{best_feature_list(1)}])
disp(['The best AUC is: ',num2str(best_AUC)])
disp('------------------------------------------')


%% Section 2.b. Select the next best feature that is best together with the first feature

[best_feature_list,best_AUC] = Add_feature(X_train,X_test,Y_train,Y_test,best_feature_list,best_AUC,'PRC');

% Add your code above this line and update the above parameters based on
% a criterion you choose to select the best features
disp(['The second best feature is number: ',num2str(best_feature_list(end)),' - ',feature_names{best_feature_list(end)}])
disp(['The best AUC is: ',num2str(best_AUC)])
disp('------------------------------------------')
% End Section 2.b.

%% Section 2.c. display selected features

% update below with graphic display, e.g. gplotmatrix

% End Section 2.c.

%% Section 4 Train ensemble models
Ensemble_bagging_MDL=[];
% update the above parameter based on your calculations
% End Section 4.

%% Section 5 display confusion matrix on test set

% use predict with Ensemble_bagging_MDL
confusion_mat=[];
% update the above parameter based on your calculations
disp('------------------------------------------')
% End Section 5.

%% Section 6 display confusion matrix on test set

Ensemble_bagging_MDL_4submission=[];
% update the above parameter based on all data
disp('------------------------------------------')
% End Section 6.
