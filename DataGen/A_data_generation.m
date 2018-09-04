clear;close all;


addpath(genpath('./.'));

%% uniform setting


scales        = 4;

%% training and testing pair
folder_train  =  {'./DIV2K_train_HR', './train291',...
    };     % training
folder_test   = {'B100'}; % testing


size_input    = 192;          % training
size_label    = 192;


stride_train  = 80;          % training
stride_test   = 80; %           % testing  %%% 

val_train     = 0;           % training % default
val_test      = 1;           % testing  % default

%% 
modelname      = ['DN_PATCH'  num2str(size_input)];
if ~exist(modelname,'file')
    mkdir(modelname);
end



%% training dataset


[data ,  labels,  set]  = patches_generation(modelname, size_input, size_label, stride_train, folder_train, val_train);






