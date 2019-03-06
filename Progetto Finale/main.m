clear
close all
clc

%% Data loading and cleaning
% The csv dataset file is imported. The first row contains the columns names
dataset = importdata('Admission_Predict_Ver1.1.csv', ',', 1);
dataset = rmfield(dataset, 'textdata'); % 'textdata' and 'colheaders' have the same data: 'textdata' is removed then
% The first column is useless, since it's just an id number. It must be dropped
dataset.data = dataset.data(:,2:end);
dataset.colheaders = dataset.colheaders(:,2:end);

%% Data normalization
% for d = 1:size(dataset.data, 2)-1
%     mi = min(dataset.data(:,d));
%     ma = max(dataset.data(:,d));
%     dataset.data(:,d) = 1/(ma-mi) * (dataset.data(:,d) - mi);
% end

%% Histogram with the distribution of chances of admission
figure
histogram(dataset.data(:,end), 10); title('Chance of Admit distribution'); % 10 buckets
%% Correlation matrix between columns
figure
R = corrcoef(dataset.data); % computation of correlation coefficients between features
h = heatmap(dataset.colheaders, dataset.colheaders, R); % heatmap of correlation coefficients
h.Title = 'Correlation matrix';
    % We can see that Chance of Admit is strongly related to GRE Score, TOEFL
    % Score and CGPA. On the other hand, Chance of Admit is poorly related to
    % Research.
%% Scatter plot between CGPA and Chance of Admit (the strongest relation in the matrix)
figure
plot(dataset.data(:,6), dataset.data(:,end), 'o', 'MarkerFaceColor', 'b');
title('CGPA-Chance of Admit'); xlabel('CGPA'); ylabel('Chance of Admit');
    % As noted before, there is a strong relation between them, since if
    % one increases also the other increases
%% Scatter plot between CGPA and Chance of Admit (the strongest relation in
% the matrix), but filtered for Research value (0 and 1)
figure, hold on
plot(dataset.data(dataset.data(:, 7) == 1, 6), dataset.data(dataset.data(:, 7) == 1, end), 'o', 'MarkerFaceColor', 'b'); % Research = 1
plot(dataset.data(dataset.data(:, 7) == 0, 6), dataset.data(dataset.data(:, 7) == 0, end), 'o', 'MarkerFaceColor', 'r'); % Research = 0
title('CGPA-Chance of Admit'); xlabel('CGPA'); ylabel('Chance of Admit');
    % We notice that almost all candidates with higher chances of admission
    % have partecipated in research work
    
%% Learning
% Application of the algorithm and visualization of the prediction function
[XT,YP,best_error,best_lambda,best_gamma] = KRLS(dataset.data(:,1:7),dataset.data(:,8),.7);
figure, hold on
plot(dataset.data(dataset.data(:, 7) == 1, 6), dataset.data(dataset.data(:, 7) == 1, end), 'o', 'MarkerFaceColor', 'b'); % Research = 1
plot(dataset.data(dataset.data(:, 7) == 0, 6), dataset.data(dataset.data(:, 7) == 0, end), 'o', 'MarkerFaceColor', 'r'); % Research = 0
plot(XT(:,6), YP, '-g', 'LineWidth', 3);
title('Chance of Admit prediction'); xlabel('CGPA'); ylabel('Chance of Admit prediction');
