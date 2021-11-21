
% CHANGE FILE NAME!!!
% REPLACE ID NUMBER with YOUR ID
clc 
clear

window_size = 20;   % Sec
over_lap = 10;      % Sec

%% Section 1.a : Iterate to load files, extract features, and build matrix
sample_rate=25;         % update according to true sample rate

d=dir('*.Acc.csv');
X=zeros(10000,48)-99;    % Allocate memory for matrix X, with default value -99
Y=zeros(10000,1)-99;     % Allocate memory for label vector Y
n_instance=0;
for r=1:length(d)
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
    
    % Moving window with over lap Predetermined
    
    n_segments=floor((length(N)/sample_rate)/over_lap)-1;
   

    for segment=1:n_segments
        ind=(segment-1)*over_lap*sample_rate+(1:(sample_rate*window_size));
        X_row=Window_features(acc_x(ind),acc_y(ind),acc_z(ind),gyro_x(ind),gyro_y(ind),gyro_z(ind));
        % replace ID number with your ID
        n_instance=n_instance+1;
        X(n_instance,:)=X_row;
        Y(n_instance)=label_segment(C,ind);
    end
end


%% Section 1.b. Features normalization/discretization remove outliers if needed
X_norm= normalize(X,1,"medianiqr");
% update the above matrices after discretization
disp('Features are after pre-processing! ')
disp('------------------------------------------')
% End Section 1.b.

%% Section 1.c. set a training & Test sets

% Devide to 70% training 30% test
train_size = size(X_norm,1);
train_ind = 1:floor(train_size*0.7);
test_ind = floor(train_size*0.7)+1:train_size;

% update the below sets
X_training=X_norm(train_ind,:);
X_test=X_norm(test_ind,:);
Y_training=Y(train_ind);
Y_test=Y(test_ind);
% End Section 1.c.

%% Section 1.d. remove correlated features

feature_names = {'max_acc_x','max_ind_acc_x','min_acc_x','min_ind_acc_x','std_acc_x','median_acc_x','bandpower_acc_x','iqr_acc_x',...
    'max_acc_y','max_ind_acc_y','min_acc_y','min_ind_acc_y','std_acc_y','median_acc_y','bandpower_acc_y','iqr_acc_y',...
    'max_acc_z','max_ind_acc_z','min_acc_z','min_ind_acc_z','std_acc_z','median_acc_z','bandpower_acc_z','iqr_acc_z',...
    'max_gyro_x','max_ind_gyro_x','min_gyro_x','min_ind_gyro_x','std_gyro_x','median_gyro_x','bandpower_gyro_x','iqr_gyro_x',...
    'max_gyro_y','max_ind_gyro_y','min_gyro_y','min_ind_gyro_y','std_gyro_y','median_gyro_y','bandpower_gyro_y','iqr_gyro_y',...
    'max_gyro_z','max_ind_gyro_z','min_gyro_z','min_ind_gyro_z','std_gyro_z','median_gyro_z','bandpower_gyro_z','iqr_gyro_z'};


% Delete corralated features
for i= 1:size(X_norm,2)

    if i<size(X_norm,2)

        R = corrcoef(X_norm);
        R = R(i,:);

        ind = (R>0.8 & R~=1);

        X_norm(:,ind) = [];
        feature_names(ind) = [];
    end
end

% R = corrcoef(X_norm);    
% 
% figure()
% heatmap(R(1:4,1:4))

%%% GPLOT - DELETE %%%

% action/no action classification
% y_tag = (Y~=0);
% 
% figure()
% gplotmatrix(X_norm(:,1:5),[],y_tag);
% 
% figure()
% gplotmatrix(X_norm(:,6:10),[],y_tag);
% 
% figure()
% gplotmatrix(X_norm(:,11:15),[],y_tag);
% 
% figure()
% gplotmatrix(X_norm(:,16:20),[],y_tag);
% 
% figure()
% gplotmatrix(X_norm(:,21:24),[],y_tag);
% 
% figure()
% gplotmatrix(X_norm(:,25:28),[],y_tag);

%%% GPLOT - DELETE %%%

% End Section 1.d.


%% Section 2.a. Select the best feature

best_AUC = 0;
best_feature=[];

for i = size(X_norm,2):-1:1
    
    train_data = X_training(:,i);
    test_data = X_test(:,i);

    model = TreeBagger(50,train_data,Y_training,'OOBPrediction','On','Method','classification');

    prediction = predict(model,test_data);

    [~,~,~,AUC] = perfcurve(Y_test,cell2mat(prediction),'0');
    
    if AUC>best_AUC
        
        best_AUC = AUC;
        best_feature = i;

    end
end


% update the above parameter based on a criterion you choose to select the
% best feature
disp(['The best features is number: ',num2str(best_feature),' - ',feature_names{best_feature}])
disp(['The best AUC is: ',num2str(best_AUC)])
disp('------------------------------------------')
% End Section 2.a.

%% Section 2.b. Select the next best feature that is best together with the first feature
best_2_features=[best_feature];

for i = 1:size(X_norm,2)
    
    train_data = X_training(:,[best_feature i]);
    test_data = X_test(:,[best_feature i]);

    model=fitensemble(train_data,Y_training,'Bag',100,'Tree','Type','classification');

    prediction = predict(model,test_data);

    [~,~,~,AUC] = perfcurve(Y_test,prediction,0);
    
    if AUC>best_AUC
        
        best_AUC = AUC;
        best_2_features(2) = i;

    end
end

% Add your code above this line and update the above parameters based on
% a criterion you choose to select the best features
disp(['The best 2 features are numbers: ',num2str(best_2_features),' - ',feature_names{best_2_features}])
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
confusion_mat=[]
% update the above parameter based on your calculations
disp('------------------------------------------')
% End Section 5.

%% Section 6 display confusion matrix on test set

Ensemble_bagging_MDL_4submission=[];
% update the above parameter based on all data
disp('------------------------------------------')
% End Section 6.
