clear
close all
clc

%% Sex Recognition

%% Data loading and normalization
dataset = 'face';
filenameX = strcat('NN_Datasets/',dataset,'_x.txt');
filenameY = strcat('NN_Datasets/',dataset,'_y.txt');
X = load(filenameX);
Y = load(filenameY);
n = size(X, 1);
percentage = .7;
nl = round(percentage * n);
indx = randperm(n);
il = sort(indx(1:nl));
iv = sort(indx(nl+1:end));
XL = X(il, :);
YL = Y(il, :);
XV = X(iv, :);
YV = Y(iv, :);
clear filenameX filenameY X Y n percentage nl indx il iv

%% Perceptron Learning Algorithm
% weights and bias initialization
w = 2 * rand(size(XL,2), 1) - 1;
b = 2 * rand(1, 1) - 1;
% compute learning error on learning set at the first step
f = (w' * XL')' + b;
err = sum(YL .* f <= 0);
i = 1;
it = 0;
% learning loop
while err > 0
    f = w' * XL(i, :)' + b;                     % prediction on learning set entry
    if YL(i) * f <= 0                           % if there's a prediction error
       w = w + YL(i) * XL(i,:)';                % update w and b
       b = b + YL(i);
       f = (w' * XL')' + b;                     % compute again the learning error
       err = sum(YL==f);
    end
    i = i + 1;
    if i > size(XL, 1)
       i = 1; 
    end
    it = it + 1;
end

%% Validation step
fV = (w' * XV')' + b;
errV = sum(YV .* fV <= 0)