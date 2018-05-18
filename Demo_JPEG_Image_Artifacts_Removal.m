
clear; %clc;
addpath(fullfile('func'));
run('E:\Tool_Box\matconvnet-1.0-beta24\matlab\vl_setupnn.m');

%% testing set
imageSets   = {'classic5','LIVE1'};
image_set   = imageSets{1};
%% model information
folderTest = fullfile('testsets', image_set);
quality_list = [10 20 30 40];
showresult  = 0;
gpu = 1;
WF = 0;
%% load model
modelName   = 'MWCNN_Haart_DBPC';
if gpu
    gpuDevice(gpu);
end

%%% read images
ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
end

%%% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));
time_all = zeros(1,length(filePaths));


for PC = quality_list % [ 10 20 30 40]
    load(fullfile('models', [modelName num2str(PC)]));

    net = dagnn.DagNN.loadobj(net) ;
    net.removeLayer('objective') ;
    out_idx = net.getVarIndex('prediction') ;
    net.vars(net.getVarIndex('prediction')).precious = 1 ;

    net.mode = 'test';

    if gpu
        net.move('gpu');
    end

    for i = 1 : length(filePaths)
        %%% read images
        im = imread(fullfile(folderTest,filePaths(i).name));
        im  = modcrop(im, 8);
        if size(im,3)==3
            label = rgb2ycbcr(im);
        else
            label = im;
        end
        sz = size(label);
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        label = im2single(label(:,:,1));


        %%
        imwrite(label, 'test.jpg','jpg','quality', PC);
        
        Compress_im = imread('test.jpg');

        delete('test.jpg');
        tic;
        output = Processing_Im(Compress_im, net, gpu, out_idx);
        times(i) = toc;

        [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),output, 0, 0);


        if showresult
            imshow(cat(2,im2uint8(input_bic),im2uint8(output),im2uint8(label)));
            title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
            drawnow;
        end
        if WF 
            path =  ['.\results\' image_set '_' modelName '_' num2str(PC)];
            if ~exist(path, 'dir'), mkdir(path) ; end
            imwrite(output, fullfile(path, [modelName '-' num2str(epoch) '-' filePaths(i).name]));
        end
        
        PSNRs(i) = PSNRCur;
        SSIMs(i) = SSIMCur;
    end
    if WF 
        save(fullfile(path, [modelName '_' image_set '_PC' num2str(PC) 'PSNR']), 'PSNRs');
        save(fullfile(path, [modelName '_' image_set '_PC' num2str(PC) 'SSIM']), 'SSIMs');
    end
    fprintf('PSNR / SSIM : %.02f / %0.4f, %0.4f.\n', mean(PSNRs),mean(SSIMs), mean(times));
end   
    
    








