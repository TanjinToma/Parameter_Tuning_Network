function [par1,par2] = parameter_tune_net(im)
%% Load the trained model and assign network weights the learned values from the trained model
%{
This example shows a network with 1 conv layer with 3 kernels (each 3×3 kernel with a added bias term), and
one dense layer 3×2 (with 3 max-pooled feature neurons and 2 output parameters. A bias term is also added for the 2 parameters)
overall, for this example, there are 38 network weights: (9×3)+3+(3×2)+2                                                
For detail understanding, see the network architecture in the paper.
%}
load('model.mat'); % load a trained model
wt=weight_vector;
% conv layer kernels 
h1=[wt(1:3)';wt(4:6)';wt(7:9)'];
h2=[wt(10:12)';wt(13:15)';wt(16:18)'];
h3=[wt(19:21)';wt(22:24)';wt(25:27)'];
% bias term to each kernel
b1=wt(28);b2=wt(29);b3=wt(30);
% dense layer including bias neuron
dense_wt=[wt(31:34)';wt(35:38)'];
% user-defined bounds of the parameters in the dense layer output. 
% Such bound denote an acceptable range of the parameter, and do not affect
% the quality of the trained model. Narrower range accelerates the training
% set the same range for both train and test 
par1_max=2.5; par1_min=0.25; 
par2_max=0.20;par2_min=0.01;
%% Network operation 
y1=conv2(abs(im),h1);y2=conv2(abs(im),h2);y3=conv2(abs(im),h3);
y1_o=poslin(y1+b1); y2_o=poslin(y2+b2); y3_o=poslin(y3+b3); % ReLU activation
f1=max(y1_o(:)); f2=max(y2_o(:)); f3=max(y3_o(:));  % max pooling
fea_in=[1.0 f1 f2 f3]';
z=dense_wt*fea_in; % dense layer
log1=1/(1+exp(-z(1))); % logistic function outputs parameter value between 0 and 1
log2=1/(1+exp(-z(2)));
par1=par1_min + (par1_max - par1_min)*log1; % using user-defined bound, scale and translate the parameter value
par2=par2_min + (par2_max - par2_min)*log2;
end

