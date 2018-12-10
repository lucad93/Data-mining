clear
close all
clc

%% Gender Recognition

%% Data loading and normalization
% load dataset
dataset = 'iris';
filenameX = strcat('NN_Datasets/',dataset,'_x.txt');
filenameY = strcat('NN_Datasets/',dataset,'_y.txt');
X = load(filenameX);
Y = load(filenameY);
% normalization of X between 0 and 1 (for iris dataset)
for i = 1:size(X,2)
   % for each feature I compute the max and min value; then I find the line s.t.
   % the normalization is 0 when value = min and 1 when value = max. I use
   % this line to normalize the dataset
   mi = min(X(:,i));                            % min value
   ma = max(X(:,i));                            % max value
   m = 1 / (ma - mi);                           % angular coeff
   q = -m * mi;                                 % constant term
   X(:,i) = m * X(:,i) + q;                     % normalization of a feature
end
% reduction of the iris classification problem to a binary classification problem
% classes 2 and 3 are considered together as a new class -1
Y(Y > 1) = -1;
% sampling of the dataset to create learning and valdation set
n = size(X, 1);                                 % number of samples
percentage = 1;
nl = round(percentage * n);                     % dimension of learning set wrt n
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
% compute learning error on learning set at the beginning
f = (w' * XL')' + b;
err = sum(YL .* f <= 0);
i = 1;                                          % entry of the dataset i'm considering
it = 0;                                         % number of iterations
% learning loop
while err > 0
    f = w' * XL(i, :)' + b;                     % prediction on learning set entry
    % if there's a prediction error
    if YL(i) * f <= 0
       % update w and b
       w = w + YL(i) * XL(i,:)';
       b = b + YL(i);
       % compute again the predictions and the learning error
       f = (w' * XL')' + b;
       err = sum(YL .* f <= 0);
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

%% Plot the 'or' and the 'iris' dataset and their linear separator
figure, hold on, grid on
separator = [-b/w(1), -b/w(2)];
% dataset points: class 1 is blue, class -1 is red
plot(XL(YL == 1, 1), XL(YL == 1, 2), 'ob');
plot(XL(YL == -1, 1), XL(YL == -1, 2), 'or');
plot([0,separator(2)],[separator(1),0],'b');    % separator line
xlim([0 1]); ylim([0 1]);                       % axes are limited between 0 and 1

%% Plot a face
% figure
% imshow(reshape(XL(1,:),[60,60])');

%% Plot w (for faces)
% figure
% imshow(reshape(w',[60,60])');