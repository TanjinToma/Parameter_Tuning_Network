function [cost] = costfun(reconstructed_output)
load('example_train_images.mat');ydata_train=train_images; % load training data
%fid = fopen('cost_values_track.txt', 'at+'); % open a file to write the
%cost values against the iterations in the Nelder-Mead optimization

for i=1:length(ydata_train)
    ref=ydata_train{i,1}; % ground-truth
    rim=reconstructed_output{i}; % reconstructed
    [aa,bb]=size(rim);
    peak_snr(i)=20*log10(sqrt(aa*bb)*1/norm(double(abs(rim))-double(abs(ref)),'fro')); % compute PSNR of the reconstructed image
    %mse_val(i)=sum(sum((rim-ref).^2))/numel(ref); % compute MSE(optional)
end
cost=-sum(peak_snr)/length(ydata_train); % compute cost
%overall_mse=sum(mse_val)/length(ydata_train);
%fprintf(fid,' %d %d \n',[cost overall_mse]); % write output in the file
%fclose(fid);
end

