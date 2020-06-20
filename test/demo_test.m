%{
Author: Tanjin Taher Toma, VITALab, University of Virginia.
Email: tt4ua@virginia.edu;
Copyright: This code is published under MIT License. See license.txt for
           more information.
Description:
This demo code shows how to estimate reconstruction regularization parameters
using our trained network [1] for a test example.

In the paper [1], we have shown the tuning of two regularization parameters of the
TLMRI reconstruction algorithm [2]. The code of the TLMRI algorithm is
available at the 'transform learning and applications software' package available at 
'http://transformlearning.csl.illinois.edu/software/'

[1] Fast Automatic Parameter Selection for MRI Reconstruction. doi: 10.1109/ISBI45749.2020.9098569.
[2] Sparsifying transform learning for Compressed Sensing MRI. doi: 10.1109/ISBI.2013.6556401.
%}

clear;clc; close all;
%% add the path of reconstruction codes
%addpath('../TLMRI'); 
%% load a Shepp-Logan Phantom image
I=phantom('Modified Shepp-Logan',240);
figure;imshow(I,[]);title('reference image');
[r,c]=size(I);
X=fftshift(fft2(I));  % frequency domian/k-space of the image
%% create a sub-sampling mask (e.g., for 2-fold subsampling, sampling along phase-encoding direction/Y axis)
Fs=2; % sampling factor
N=30; % number of center lines to fill in the mask
Ns=floor(r/Fs); % total lines to sample
Q=zeros(size(I)); 
Q(floor(r/2)-floor(N/2)+1:floor(r/2)+ceil(N/2),:)=1; ind=find(Q(:,1)==1);
N_lines=Ns-length(ind);
S=setdiff(1:r,ind);indr=sort(randsample(S,N_lines,false),'ascend'); 
Q(indr,:)=1;
figure;imshow(Q,[]);title('sub-sampling mask');
%% subsample the image in k-space and add noise to simulate noisy subsampled k-space. 
% Next, generate the artifact image by inverse Fourier transform
snr=35; % noise snr
usampled_kspace=X.*Q;
col_data=reshape(usampled_kspace,[numel(usampled_kspace),1]);
sigma = norm(col_data)/sqrt(numel(usampled_kspace))/10^(snr/20); % noise std
usampled_kspace = usampled_kspace + (sigma/sqrt(2)).*complex(randn(size(usampled_kspace)),randn(size(usampled_kspace)));
usampled_im=ifft2(usampled_kspace);
figure;imshow(abs(usampled_im),[]);title('sub-sampled input image'); 
%% with subsampled image, estimate reconstruction parameters using parameter tuning network (e.g., two TLMRI parameters shown here)
[par1,par2]=parameter_tune_net(usampled_im)

%% reconstruct the image from noisy subsampled k-space using estimated parameters
%[recon_img,~]= TLMRI_recon(usampled_kspace,Q,par1,par2); 
% TLMRI reconstruction code is available at 
% figure;imshow(recon_img,[]);
%% compute PSNR and SSIM metrics on the reconstructed image quality
% [aa,bb]=size(recon_img);
% peak_snr=20*log10(sqrt(aa*bb)*1/norm(double(abs(recon_img))-double(abs(I)),'fro'))
% ssim_index=ssim(double(abs(recon_img)), double(abs(I)))

