%{
Author: Tanjin Taher Toma, VITALab, University of Virginia.
Email: tt4ua@virginia.edu;
Copyright: This code is published under MIT License. See license.txt for
           more information.
Description:
This code shows how to train our parameter tuning network [1] to estimate
regularization parameters of a certain reconstruction algorithm. 

In the paper [1], we have shown the tuning of two regularization parameters of the
TLMRI reconstruction algorithm [2]. The code of the TLMRI algorithm is
available at the 'transform learning and applications software' package available at 
'http://transformlearning.csl.illinois.edu/software/. 

Running the 'main_train.m' script outputs the trained model on a certain training dataset

The 'example_train_images.mat' file shows how we have
organized the training data in a cell array. The first column in the
array is the reference image, second column is the subsampling mask, third
column is the SNR value of the input (in measurement domain).

[1] Fast Automatic Parameter Selection for MRI Reconstruction. doi: 10.1109/ISBI45749.2020.9098569.
[2] Sparsifying transform learning for Compressed Sensing MRI. doi: 10.1109/ISBI.2013.6556401.
%}

clear;clc; close all;
%% % add the path of the reconstruction algorithm
addpath('../TLMRI'); % put TLMRI reconstruction function from TL softwate package inside this directory
%% initialize network weights
%{
This example shows a network with 1 conv layer with 3 kernels (each 3×3 kernel with a added bias term), and
one dense layer 3×2 (with 3 max-pooled feature neurons and 2 output parameters. A bias term is also added for the 2 parameters)
overall, there are 38 network weights: (9×3)+3+(3×2)+2                                                
For detail understanding, see the network architecture in the paper.
%}
initialsimplex = 0 + (1)*rand(38,38+1); % initial simplex for Nelder-Mead
%% Run the Nelder-Mead optimization to find the unknown weights of parameter tuning network
reconfun=@param_estimate_recon; % function for parameter selection and reconstruction of the training examples
optfun=@costfun; % function to compute cost
% specify Nelder-Mead parameters
NM.niters=1; %  iteration number
NM.tolfun=1e-4; % a tolernace threshold
NM.tolparams=1e-4; % a tolernace threshold
[weight_vector,optval,info,varargout] = Nelder_Mead_optimization(initialsimplex,[],[],reconfun,optfun,NM);

save('model.mat','weight_vector'); % save the trained model having the network weights



