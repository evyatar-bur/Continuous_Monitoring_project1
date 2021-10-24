function [t,x,y,z] = read_data(file_name)
% read_data function reads the given file and returns the following vectors:
% time and x,y,z measurments 

M = csvread(file_name,1,2);

t = M(:,1);
x = M(:,2);
y = M(:,3);
z = M(:,4);

end

