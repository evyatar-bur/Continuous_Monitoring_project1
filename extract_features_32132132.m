function X_row=extract_features_32132132(acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z)

X_row=zeros(1,30);
% Write code to update the above X_row with 30 features
X_row(1)=mean(acc_x);
X_row(2)=std(acc_y);
X_row(3)=std(acc_z)-std(acc_x);
X_row(4)=mean(gyro_x);
X_row(5)=std(gyro_y);
X_row(6)=std(gyro_z)-std(gyro_x);
% etc....