function [reconstructed_imgs] = param_estimate_recon(wt)
%addpath('./TLMRI'); 
load('example_train_images.mat');xdata_train=train_images; % load training data

%% define the network architecture
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
%% parameter selection and reconstruction for all training examples
reconstructed_imgs={};
parfor i=1:length(xdata_train)
    I=xdata_train{i,1}; % reference image
    X=fftshift(fft2(I));% frequency domian of the original image
    Q=xdata_train{i,2};% sub-sampling mask
    snr=xdata_train{i,3}; % snr of the input kspace
    usampled_kdata=X.*Q; % undersampled k-space
    col_data=reshape(usampled_kdata,[numel(usampled_kdata),1]);
    sigma = norm(col_data)/sqrt(numel(usampled_kdata))/10^(snr/20);
    usampled_kdata = usampled_kdata + (sigma/sqrt(2)).*complex(randn(size(usampled_kdata)),randn(size(usampled_kdata))); % add noise
    usampled_im=ifft2(usampled_kdata); % under-sampled/zero-filled image
    %% network takes each zero-filled image and estimates reconstruction parameters
    y1=conv2(abs(usampled_im),h1);y2=conv2(abs(usampled_im),h2);y3=conv2(abs(usampled_im),h3);
    y1_o=poslin(y1+b1); y2_o=poslin(y2+b2); y3_o=poslin(y3+b3); % ReLU activation
    f1=max(y1_o(:)); f2=max(y2_o(:)); f3=max(y3_o(:));  % max pooling
    fea_in=[1.0 f1 f2 f3]';
    z=dense_wt*fea_in; % dense layer
    log1=1/(1+exp(-z(1))); % logistic function outputs parameter value between 0 and 1
    log2=1/(1+exp(-z(2)));
    par1=par1_min + (par1_max - par1_min)*log1; % using user-defined bound, scale and translate the parameter value
    par2=par2_min + (par2_max - par2_min)*log2;
    %% reconstructing the image from noisy subsampled k-space using estimated parameters
    [recon_img,~]= TLMRI_recon(usampled_kdata,Q,par1,par2); % put the 'TLMRI_recon' function in the 'TLMRI' directory
    reconstructed_imgs{i}=recon_img;
end
end

