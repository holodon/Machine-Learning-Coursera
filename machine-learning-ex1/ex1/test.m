%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


theta = zeros(size(X(1,:)))'; % initialize fitting parameters
alpha = 1.1; %% Your initial learning rate %%
J1 = zeros(50, 1); 

for num_iterations = 1:50
    J1(num_iterations) = (X*theta - y)'*(X*theta - y)/(2*m); %% Calculate your cost function here %%
    theta = theta - alpha * (1/m) * (((X * theta) - y)' * X)'; %% Result of gradient descent update %%
end

% now plot J
% technically, the first J starts at the zero-eth iteration
% but Matlab/Octave doesn't have a zero index
figure;
plot(0:49, J1(1:50), 'b-')
xlabel('Number of iterations')
ylabel('Cost J')
