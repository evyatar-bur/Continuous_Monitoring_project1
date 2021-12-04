function [confusion_mat] = main(path)

window_size = 16;   % Sec

max_last_window=ones(1,6); % for first window features calc

% Suppress readtable warning
warning('off','MATLAB:table:ModifiedAndSavedVarnames')

%% Section 1.a : Iterate to load files, extract features, and build matrix
sample_rate = 25;      

% Change current directory to path
cd(path)

% Read recordings
d=dir('*.Acc.csv');

X_event=zeros(50000,10)-99;    % Allocate memory for matrix X, with default value -99
Y_event=zeros(50000,1)-99;     % Allocate memory for label vector Y

n_instance=0;                  % Window counter

% make a High Pass Filter
fco = 0.1;                     % cutoff frequency (Hz)
Np = 2;                        % filter order=number of poles

[b,a]=butter(Np,fco/(sample_rate/2),'high'); 

for r=1:length(d)

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
    
    % Find suspected events for window
    [~,locs] = findpeaks(gyro_x,'MinPeakHeight',15,'MinPeakDistance',250);  
    
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
            X_row = extract_selected_features_204613681_308317361(acc_x(ind),acc_y(ind),acc_z(ind),gyro_x(ind),gyro_y(ind),gyro_z(ind),max_last_window);
            
            n_instance=n_instance+1;
            
            % Add feature vector to feature matrix
            X_event(n_instance,:) = X_row;
            Y_event(n_instance) = label_segment(C,ind,N);

        end
    end
end

% Delete empty rows
ind = find(Y_event~=-99);
X_event = X_event(ind,:);
Y_event = Y_event(ind,:);

%% Section 1.b. Features normalization
X_norm = normalize(X_event,1,'medianiqr');

disp('------------------------------------------')
disp('Features are after pre-processing! ')
disp('------------------------------------------')
% End Section 1.b.

%% Load trained event trigger classifier

model_struct = load('Event_trigger_model.mat');
model = model_struct.Ensemble_bagging_MDL_4submission;

% Predict labels using trained model
prediction = predict(model,X_norm);

% Create confusion matrix
confusion_mat = confusionmat(Y_event,prediction);

% Combine horizontal and vertical zoom 
confusion_mat(8,:) = confusion_mat(8,:) + confusion_mat(9,:);
confusion_mat(:,8) = confusion_mat(:,8) + confusion_mat(:,9);
confusion_mat(:,9) = [];
confusion_mat(9,:) = [];

confusion_mat(6,:) = confusion_mat(6,:) + confusion_mat(7,:);
confusion_mat(:,6) = confusion_mat(:,6) + confusion_mat(:,7);
confusion_mat(:,7) = [];
confusion_mat(7,:) = [];


% End Section 5.

end