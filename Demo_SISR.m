
clear; %clc;


run('E:\Tool_Box\matconvnet-1.0-beta24\matlab\vl_setupnn.m');
addpath(fullfile('func'));

%% testing set
imageSets   = {'Set5','Set14','B100','Urban100'};
image_set   = imageSets{1};
%% model information
folderTest = fullfile('testsets','Test',image_set);

showresult  = 0;
scalelist      = [2 3 4];

gpu = 1;
if gpu > 0
    gpuDevice(gpu);
end
WF = 0;

modelName   = 'MWCNN_Haart_SRx';

%%% read images
ext         =  {'*.bmp','*.png','*.jpg'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
end

%%% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));
times = zeros(1,length(filePaths));



for scale = scalelist 
    load(fullfile('models', [modelName num2str(scale)]));
%     load('E:\MyCode\Wavelet\MWCNN_MODEL\data\MWCNN_Haart_SRx3_L3L24_F160_256_256.mat')
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
        im = imread(fullfile(folderTest, filePaths(i).name));
        if scale == 3 
            im  = modcrop(im, 8*scale);
        else
            im  = modcrop(im, 8);
        end
        if size(im,3)==3
            label = rgb2ycbcr(im);
        else
            label = im;
        end

        sz = size(label);
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        output_ycbcr =  imresize(imresize(label, 1/scale,'bicubic'),scale,'bicubic');
        label = im2single(label(:,:,1));
        %%
        
        LR = imresize(imresize(label,1/scale,'bicubic'),scale,'bicubic');
        tic;
        output = Processing_Im(LR, net, gpu, out_idx);
        times(i) = toc;

        if size(im,3)==3
            output = im2uint8(output);
        end
        %%% calculate PSNR and SSIM
        [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output), 0, 0);

        if showresult
            imshow(cat(2,im2uint8(input_bic),im2uint8(output),im2uint8(label)));
            title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
            drawnow;
        end
        if WF 
            %%% Save RGB and y 
            path =  ['.\results\' image_set '_' modelName '_x' num2str(scale)];
            if ~exist(path, 'dir'), mkdir(path) ; end
            imwrite(output, fullfile(path, [modelName '-' num2str(epoch) '-' filePaths(i).name]));
            path =  ['.\results\' image_set '_' modelName '_x' num2str(scale) '_RGB'];
            if ~exist(path, 'dir'), mkdir(path) ; end
            if size(im,3)==3
                output_ycbcr(:,:,1) = output;
                imwrite(ycbcr2rgb(output_ycbcr), fullfile(path, [modelName '_' num2str(epoch) '_RGB_' filePaths(i).name]));
            else
                imwrite(output, fullfile(path, [modelName '_' num2str(epoch) '_' filePaths(i).name]));
            end
        end
        
        PSNRs(i) = PSNRCur;
        SSIMs(i) = SSIMCur;
    end
    if WF 
        save(fullfile(path, [modelName '_' image_set  '_x' num2str(scale) 'PSNR']), 'PSNRs');
        save(fullfile(path, [modelName '_' image_set  '_x' num2str(scale) 'SSIM']), 'SSIMs');
    end
    fprintf('PSNR / SSIM : %.02f / %0.4f, %0.4f.\n', mean(PSNRs),mean(SSIMs), mean(times));
end   
        





