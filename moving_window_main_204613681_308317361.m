
clc 
clear

window_size = 16;   % Sec
over_lap = 8;      % Sec

% Suppress readtable warning
warning('off','MATLAB:table:ModifiedAndSavedVarnames')

%% Section 1.a : Iterate to load files, extract features, and build matrix
sample_rate=25;      

d=dir('*.Acc.csv');
X=zeros(50000,72)-99;    % Allocate memory for matrix X, with default value -99
Y=zeros(50000,1)-99;     % Allocate memory for label vector Y

n_instance = 0; % Window counter
max_last_window=ones(1,6); %for first window features calc


% make a filter and apply it to the signal
fco = 0.1;          % cutoff frequency (Hz)
Np = 2;           % filter order=number of poles

[b,a]=butter(Np,fco/(sample_rate/2),'high'); %high pass Butterworth filter coefficients

cut_flag = true;


for r=1:length(d)

    disp(d(r).name)

    if contains(d(r).name, '25') && cut_flag
        
        cut_flag = false;
        cut_ind = n_instance;
    end

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
    
    % apply the filter
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

    % Moving window with over lap Predetermined
    n_segments=floor((N/sample_rate)/over_lap)-1;
    
    % Compute features for each window
    for segment=1:n_segments
        ind=(segment-1)*over_lap*sample_rate+(1:(sample_rate*window_size));
        X_row = extract_features_204613681_308317361(acc_x(ind),acc_y(ind),acc_z(ind),gyro_x(ind),gyro_y(ind),gyro_z(ind),max_last_window);
        max_last_window=X_row([1 13 25 37 49 61]);
        n_instance=n_instance+1;

        X(n_instance,:) = X_row;
        Y(n_instance) = label_segment(C,ind,N);
    end
end

% Delete empty rows
ind = find(Y~=-99);
X = X(ind,:);
Y = Y(ind,:);

%% Section 1.b. Features normalization/discretization remove outliers if needed
X_norm = normalize(X,1,'medianiqr');   % DELETE!!! ,range',[0 1]); DELETE!!!
% update the above matrices after discretization
disp('------------------------------------------')
disp('Features are after pre-processing! ')
disp('------------------------------------------')
% End Section 1.b.

%% Section 1.c. set training & Test sets

% Devide data to test and train - 8 last records are test data

% update the below sets
X_train=X_norm(1:cut_ind,:);   % 34029 for 20 sec window/10 sec overlap
X_test=X_norm(cut_ind+1:end,:);
Y_train=Y(1:cut_ind);
Y_test=Y(cut_ind+1:end);

% End Section 1.c.

%% Section 1.d. remove correlated features

feature_names = {'max acc x','zero cross acc x','min acc x','diff acc x','std acc x','median acc x','bandpower acc x','mean squared acc x','skewness a x','max/last window acc x','tresh 25% acc x','tresh const acc x',...
    'max acc y','zero cross acc y','min acc y','diff acc y','std acc y','median acc y','bandpower acc y','mean squared acc y','skewness a y','max/last window acc y','tresh 25% acc y','tresh const acc y',...
    'max acc z','zero cross acc z','min acc z','diff acc z','std acc z','median acc z','bandpower acc z','mean squared acc z','skewness a z','max/last window acc z','tresh 25% acc z','tresh const acc z',...
    'max gyro x','zero cross gyro x','min gyro x','diff gyro x','std gyro x','median gyro x','bandpower gyro x','mean squared gyro x','skewness g x','max/last window gyro x','tresh 25% gyro x','tresh const gyro x',...
    'max gyro y','zero cross gyro y','min gyro y','diff gyro y','std gyro y','median gyro y','bandpower gyro y','mean squared gyro y','skewness g y','max/last window gyro y','tresh 25% gyro y','tresh const gyro y',...
    'max gyro z','zero cross gyro z','min gyro z','diff gyro z','std gyro z','median gyro z','bandpower gyro z','mean squared gyro z','skewness g z','max/last window gyro z','tresh 25% gyro z','tresh const gyro z'};


Y_train_hat = (Y_train ~= 0);

len = size(X_train,2);
W = zeros(len,1);
for j=1:len
    [~,W(j)] = relieff(X_train(:,j),Y_train,10);
end

[~,ind] = sort(W,'descend');

X_train = X_train(:,ind);
X_test = X_test(:,ind);
X_norm = X_norm(:,ind);
feature_names = feature_names(ind);
W = W(ind);

% Delete corralated features
for i = 1:size(X_train,2)

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
best_AUC = 0;
method = 'F1'; % 'F1', 'ROC' or 'PRC'

[best_feature_list,best_AUC] = Add_feature(X_train,X_test,Y_train,Y_test,best_feature_list,best_AUC,method);

% update the above parameter based on a criterion you choose to select the
% best feature
disp(['The best feature is number: ',num2str(best_feature_list(1)),' - ',feature_names{best_feature_list(1)}])
disp(['The best AUC is: ',num2str(best_AUC)])
disp('------------------------------------------')


%% Section 2.b. Select the next best feature that is best together with the first feature

[best_feature_list,best_AUC] = Add_feature(X_train,X_test,Y_train,Y_test,best_feature_list,best_AUC,method);

% Add your code above this line and update the above parameters based on
% a criterion you choose to select the best features
disp(['The second best feature is number: ',num2str(best_feature_list(end)),' - ',feature_names{best_feature_list(end)}])
disp(['The best AUC is: ',num2str(best_AUC)])
disp('------------------------------------------')

%% Add more features

for i = 1:4

    [best_feature_list,best_AUC] = Add_feature(X_train,X_test,Y_train,Y_test,best_feature_list,best_AUC,method);
    disp(['The new best feature is number: ',num2str(best_feature_list(end)),' - ',feature_names{best_feature_list(end)}])
    disp(['The best AUC is: ',num2str(best_AUC)])
    disp('------------------------------------------')

end

% End Section 2.b.

%% Section 2.c. display selected features

% update below with graphic display, e.g. gplotmatrix

% End Section 2.c.

%% Section 4 Train ensemble models

train_data = X_train(:,best_feature_list);
test_data = X_test(:,best_feature_list);

t = templateTree('MaxNumSplits',100);

Ensemble_bagging_MDL=fitcensemble(train_data,Y_train,'method','RUSBoost','NumLearningCycles',100,'Learners',t,'LearnRate',0.1);

% update the above parameter based on your calculations
% End Section 4.

%% Section 5 display confusion matrix on test set

% Predict scores and predictions
[prediction,scores] = predict(Ensemble_bagging_MDL,test_data);

% use predict with Ensemble_bagging_MDL
[confusion_mat,~] = confusionmat(Y_test,prediction)

figure()
confusionchart(Y_test,prediction,'Normalization','row-normalized','RowSummary','row-normalized')
% update the above parameter based on your calculations
disp('------------------------------------------')
% End Section 5.

 %% Section 6 - Create final model for submission
Final_data = X_norm(:,best_feature_list);

Ensemble_bagging_MDL_4submission= fitcensemble(train_data,Y_train,'method','RUSBoost','NumLearningCycles',1000,'Learners',t,'LearnRate',0.1);
% update the above parameter based on all data
disp('------------------------------------------')
% End Section 6.

%% Visualization 

% close all
% 
% % Gplotmatrix - all used features
% figure()
% gplotmatrix(X_norm,[],Y)
% title('Gplotmatrix - features used in model')
% 
% % Gplotmatrix - 2 best features
% figure()
% gplotmatrix(X_norm(:,best_feature_list(1:2)),[], Y,[],[],[],[],[],feature_names(best_feature_list(1:2)))
% title('Gplotmatrix - 2 best features')

