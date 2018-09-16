
%%% Note: run the 'DataGen/A_data_generation.m' to generate
%%% training data first.
clear;
rng('default')
addpath(genpath('./.'));
addpath(genpath('../.'));
addpath(genpath('../DataGen'));
addpath('matconvnet');
vl_setupnn;

%%%-------------------------------------------------------------------------
%%% Data generation
%%%-------------------------------------------------------------------------
A_data_generation;

%%%-------------------------------------------------------------------------
%%% Configuration
%%%-------------------------------------------------------------------------

opts.gpus             = [1 2]; %%% this code can only support multi-GPU!



opts.numSubBatches    = 1;
opts.bnormLearningRate= 0;

%%% solver
opts.solver           = 'Adam';
opts.numberImdb       = 1;

opts.derOutputs       = {'objective', 1} ;

ext               =  {'*.jpg','*.png','*.bmp'};

opts.gradientClipping = false; %%% set 'true' to prevent exploding gradients in the beginning.
opts.backPropDepth    = Inf;
%%%-------------------------------------------------------------------------
%%%   Initialize model and load data
%%%---------------------------------------------------- 

%%%  load data

global CurTask;
CurTask = 'Denoising'; %% 'Deblocking' and 'SISR'


opts.imdbDir          = '../DataGen/DN_PATCH192';
filepaths           =  [];
for i = 1 : length(ext)
    filepaths = [filepaths; dir(fullfile(opts.imdbDir, ext{i}))];
end
imdb.imdbPath = opts.imdbDir;
imdb.filepaths = filepaths;
imdb.images.set = ones(numel(filepaths),1);
fprintf('-----------------------------------------------------------\n');
fprintf('--------------------Training Number %d---------------------\n', numel(filepaths));
fprintf('-----------------------------------------------------------\n');
image = imread(fullfile(opts.imdbDir, filepaths(1).name));
imdb.patch_size = size(image, 1);


opts.learningRate     = [logspace(-3, -3, 15) logspace(-3.8, -4, 20) logspace(-4.5, -5, 10)];

%% for denosining, set simga to [15, 25, 50]; for SISR, set simga to [2, 3, 4], and for deblocking, set sigma to [10, 20, 30, 40]
opts.sigma            = 50; 

opts.modelName        = ['MWCNN_GDSigma' num2str(opts.sigma)]; %%% model name

opts.expDir      = fullfile('data', opts.modelName);
opts.batchSize        = 32*numel(opts.gpus);
net = net_wavelet_haart_24;


[~, ~] = cnn_train(net, imdb, ...
    'expDir', opts.expDir, ...
    'learningRate',opts.learningRate, ...
    'derOutputs',opts.derOutputs, ...
    'bnormLearningRate',opts.bnormLearningRate, ...
    'numSubBatches',opts.numSubBatches, ...
    'numberImdb',opts.numberImdb, ...
    'backPropDepth',opts.backPropDepth, ...
    'batchSize', opts.batchSize, ...
    'modelname', opts.modelName, ...
    'sigma', opts.sigma, ... 
    'gpus',opts.gpus) ;

