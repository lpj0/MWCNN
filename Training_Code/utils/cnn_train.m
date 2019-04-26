 function [net,stats] = cnn_train(net, imdb, varargin)
%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
opts.modelName   = 'model';
opts.expDir = fullfile('data',opts.modelName) ;
opts.continue = true ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 20 ;

%%% solver: Adam
%%% opts.solver = 'Adam';
opts.beta1   = 0.9;
opts.beta2   = 0.999;
opts.alpha   = 0.01;
opts.epsilon = 1e-8;


%%% solver: SGD
opts.solver = 'SGD';
opts.learningRate = 0.001;
opts.weightDecay  = 0.0005;
opts.momentum     = 0.9 ;

%%% GradientClipping
opts.gradientClipping = false;
opts.theta            = 0.005;

%%% specific parameter for Bnorm
opts.bnormLearningRate = 0;

opts.randomSeed = 0 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;

opts.derOutputs = {'objective', 1} ;
opts.extractStatsFn = @extractStats ;
opts.plotStatistics = true;
opts.conserveMemory = true;
opts.mode = 'normal';
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false;
opts.numSubBatches = 1;
opts.numberImdb   = 1;
opts.sigma = 50;

opts = vl_argparse(opts, varargin) ;
opts.numEpochs = numel(opts.learningRate);


if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  if isempty(opts.derOutputs)
    error('DEROUTPUTS must be specified when training.\n') ;
  end
end


state.getBatch = getBatch ;
stats = [] ;

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf([opts.modelName '-epoch-%d.mat'], ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir,opts.modelName);
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
  [net, stats] = loadState(modelPath(start)) ;
  
%   net.params(8).learningrate = update_flag;
%   net.params(19).learningrate = update_flag;
%   net.params(60).learningrate = update_flag;
%   net.params(71).learningrate = update_flag;
end

for epoch=start+1:opts.numEpochs
%   parpool;
  % Set the random seed based on the epoch and opts.randomSeed.
  % This is important for reproducibility, including when training
  % is restarted from a checkpoint.

  rng(epoch + opts.randomSeed) ;
  prepareGPUs(opts, epoch == start+1) ;

  % Train for one epoch.

  state.epoch = epoch ;
  state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  state.thetaCurrent = opts.theta(min(epoch, numel(opts.theta)));
  state.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  state.val = opts.val(randperm(numel(opts.val))) ;
  
  state.beta1 = opts.beta1;
  state.beta2 = opts.beta2;
  state.alpha = opts.alpha;
  state.epsilon = opts.epsilon;
  state.gradientClipping = opts.gradientClipping;
  
  state.imdb = imdb ;
  
%   state = distributed(state);

  if numel(opts.gpus) <= 1
    [stats.train(epoch),prof] = process_epoch(net, state, opts, 'train') ;
    stats.val(epoch) = process_epoch(net, state, opts, 'val') ;
    if opts.profile
      profview(0,prof) ;
      keyboard ;
    end
  else
    savedNet = net.saveobj() ;
    spmd
      net_ = dagnn.DagNN.loadobj(savedNet) ;
      [stats_.train, prof_] = process_epoch(net_, state, opts, 'train') ;
      stats_.val = process_epoch(net_, state, opts, 'val') ;
      if labindex == 1, savedNet_ = net_.saveobj() ; end
    end
    net = dagnn.DagNN.loadobj(savedNet_{1}) ;
    stats__ = accumulateStats(stats_) ;
    stats.train(epoch) = stats__.train ;
    stats.val(epoch) = stats__.val ;
    if opts.profile
      mpiprofile('viewer', [prof_{:,1}]) ;
      keyboard ;
    end
    clear net_ stats_ stats__ savedNet savedNet_ ;
  end

  % save
  if ~evaluateMode
    saveState(modelPath(epoch), net, stats) ;
  end

  if opts.plotStatistics
    switchFigure(1) ; clf ;
    plots = setdiff(...
      cat(2,...
      fieldnames(stats.train)', ...
      fieldnames(stats.val)'), {'num', 'time'}) ;
    for p = plots
      p = char(p) ;
      values = zeros(0, epoch) ;
      leg = {} ;
      for f = {'train', 'val'}
        f = char(f) ;
        if isfield(stats.(f), p)
          tmp = [stats.(f).(p)] ;
          values(end+1,:) = tmp(1,:)' ;
          leg{end+1} = f ;
        end
      end
      subplot(1,numel(plots),find(strcmp(p,plots))) ;
      plot(1:epoch, values','o-') ;
      xlabel('epoch') ;
      title(p) ;
      legend(leg{:}) ;
      grid on ;
    end
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
  end
%   delete(gcp('nocreate'));
end

% -------------------------------------------------------------------------
function [stats, prof] = process_epoch(net, state, opts, mode)
% -------------------------------------------------------------------------

% initialize empty momentum
if strcmp(mode,'train')
  state.momentum = num2cell(zeros(1, numel(net.params))) ;
  state.m = num2cell(zeros(1, numel(net.params))) ;
  state.v = num2cell(zeros(1, numel(net.params))) ;
  state.t = num2cell(zeros(1, numel(net.params))) ;
end

% move CNN  to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
  if strcmp(mode,'train')
    state.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
    state.m = cellfun(@gpuArray,state.m,'UniformOutput',false) ;
    state.v = cellfun(@gpuArray,state.v,'UniformOutput',false) ;
    state.t = cellfun(@gpuArray,state.t,'UniformOutput',false) ;
  end
end
if numGpus > 1
  mmap = map_gradients(opts.memoryMapFile, net, numGpus) ;
else
  mmap = [] ;
end

% profile
if opts.profile
  if numGpus <= 1
    profile clear ;
    profile on ;
  else
    mpiprofile reset ;
    mpiprofile on ;
  end
end

subset = state.(mode) ;
num = 0 ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;
diary(fullfile(opts.expDir, [opts.modelName '_test.txt']));
diary on;
start = tic ;
for t=1:opts.batchSize:numel(subset)
  fprintf('%s %s: epoch %02d: %3d/%3d:', opts.modelName, mode, state.epoch, ...
          fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;

  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    inputs = state.getBatch(state.imdb, batch, opts.sigma) ;

    if opts.prefetch
      if s == opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      state.getBatch(state.imdb, nextBatch) ;
    end

    if strcmp(mode, 'train')
      net.mode = 'normal' ;
      net.accumulateParamDers = (s ~= 1) ;
      net.eval(inputs, opts.derOutputs) ;
    else
      net.mode = 'test' ;
      net.eval(inputs) ;
    end
  end

  % accumulate gradient
  if strcmp(mode, 'train')
    if ~isempty(mmap)
      write_gradients(mmap, net) ;
      labBarrier() ;
    end
    state = accumulate_gradients(state, net, opts, batchSize, mmap) ;
  end

  % get statistics
  time = toc(start) + adjustTime ;
  batchTime = time - stats.time ;
  stats = opts.extractStatsFn(net) ;
  stats.num = num ;
  stats.time = time ;
  currentSpeed = batchSize / batchTime ;
  averageSpeed = (t + batchSize - 1) / time ;
  if t == opts.batchSize + 1
    % compensate for the first iteration, which is an outlier
    adjustTime = 2*batchTime - time ;
    stats.time = time + adjustTime ;
  end

  fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
  for f = setdiff(fieldnames(stats)', {'num', 'time'})
    f = char(f) ;
    fprintf(' %s:', f) ;
    fprintf(' %.3f', stats.(f)) ;
  end
  fprintf('\n') ;
end
diary off;
if ~isempty(mmap)
  unmap_gradients(mmap) ;
end

if opts.profile
  if numGpus <= 1
    prof = profile('info') ;
    profile off ;
  else
    prof = mpiprofile('info');
    mpiprofile off ;
  end
else
  prof = [] ;
end

net.reset() ;
net.move('cpu') ;

% -------------------------------------------------------------------------
function state = accumulate_gradients(state, net, opts, batchSize, mmap)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;

for p=1:numel(net.params)

  % accumualte gradients from multiple labs (GPUs) if needed
  if numGpus > 1
    tag = net.params(p).name ;
    for g = otherGpus
      tmp = gpuArray(mmap.Data(g).(tag)) ;
      net.params(p).der = net.params(p).der + tmp ;
    end
  end

  switch net.params(p).trainMethod

    case 'average' % mainly for batch normalization
%       thisLR = net.params(p).learningRate ;
%       net.params(p).value = ...
%           (1 - thisLR) * net.params(p).value + ...
%           (thisLR/batchSize/net.params(p).fanout) * net.params(p).der ;
      thisLR = net.params(p).learningRate ;
      net.params(p).value = vl_taccum(...
          1 - thisLR, net.params(p).value, ...
          (thisLR/batchSize/net.params(p).fanout),  net.params(p).der) ;

    case 'gradient'
        thisLR = state.learningRate * net.params(p).learningRate ;
        state.t{p} = state.t{p} + 1;
        t = state.t{p};
        alpha = thisLR; % opts.alpha; %opts.alpha / sqrt(t);
        lr = alpha * sqrt(1 - state.beta2^t) / (1 - state.beta1^t);
        if lr > 0 && thisLR > 0
            state.m{p} = vl_taccum( state.beta1, state.m{p}, 1 - state.beta1, net.params(p).der); 
            state.v{p} = vl_taccum( state.beta2, state.v{p}, 1 - state.beta2, (net.params(p).der).^2); 
            net.params(p).value = vl_taccum(1, net.params(p).value, -gather(lr), state.m{p} ./ (state.v{p}.^0.5 + state.epsilon));
        end
      
    case 'adam'
        thisLR = state.learningRate * net.params(p).learningRate ;
        state.t{p} = state.t{p} + 1;
        t = state.t{p};
        alpha = thisLR; % opts.alpha; %opts.alpha / sqrt(t);
        lr = alpha * sqrt(1 - state.beta2^t) / (1 - state.beta1^t);
        if lr > 0 && thisLR > 0
            state.m{p} = vl_taccum( state.beta1, state.m{p}, 1 - state.beta1, net.params(p).der); 
            state.v{p} = vl_taccum( state.beta2, state.v{p}, 1 - state.beta2, (net.params(p).der).^2); 
            net.params(p).value = vl_taccum(1, net.params(p).value, -gather(lr), state.m{p} ./ (state.v{p}.^0.5 + state.epsilon));
        end

    case 'otherwise'
      error('Unknown training method ''%s'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.params)
  format(end+1,1:3) = {'single', size(net.params(i).value), net.params(i).name} ;
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, ...
                  'Format', format, ...
                  'Repeat', numGpus, ...
                  'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net)
% -------------------------------------------------------------------------
for i=1:numel(net.params)
  mmap.Data(labindex).(net.params(i).name) = gather(net.params(i).der) ;
end

% -------------------------------------------------------------------------
function unmap_gradients(mmap)
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

for s = {'train', 'val'}
  s = char(s) ;
  total = 0 ;

  % initialize stats stucture with same fields and same order as
  % stats_{1}
  stats__ = stats_{1} ;
  names = fieldnames(stats__.(s))' ;
  values = zeros(1, numel(names)) ;
  fields = cat(1, names, num2cell(values)) ;
  stats.(s) = struct(fields{:}) ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
stats = struct() ;
for i = 1:numel(sel)
  stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net, stats)
% -------------------------------------------------------------------------
net_ = net ;
net = net_.saveobj() ;
save(fileName, 'net', 'stats') ;

% -------------------------------------------------------------------------
function [net, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir, modelName)
% -------------------------------------------------------------------------
% list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
% tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
% epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
% epoch = max([epoch 0]) ;
list = dir(fullfile(modelDir, [modelName,'-epoch-*.mat'])) ;
tokens = regexp({list.name}, [modelName,'-epoch-([\d]+).mat'], 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

% -------------------------------------------------------------------------
function switchFigure(n)
% -------------------------------------------------------------------------
if get(0,'CurrentFigure') ~= n
  try
    set(0,'CurrentFigure',n) ;
  catch
    figure(n) ;
  end
end

% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
if numGpus > 1
  % check parallel pool integrity as it could have timed out
  pool = gcp('nocreate') ;
  if ~isempty(pool) && pool.NumWorkers ~= numGpus
    delete(pool) ;
  end
  pool = gcp('nocreate') ;
  if isempty(pool)
    parpool('local', numGpus) ;
    cold = true ;
  end
  if exist(opts.memoryMapFile)
    delete(opts.memoryMapFile) ;
  end

end
if numGpus >= 1 && cold
  fprintf('%s: resetting GPU\n', mfilename)
  if numGpus == 1
    gpuDevice(opts.gpus)
  else
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
end

%end
%%%-------------------------------------------------------------------------
function A = gradientClipping(A, theta,gradientClip)
%%%-------------------------------------------------------------------------
if gradientClip
    A(A>theta)  = theta;
    A(A<-theta) = -theta;
else
    return;
end

% -------------------------------------------------------------------------
function fn = getBatch()
% -------------------------------------------------------------------------
fn = @(x,y,z) getDagNNBatch(x,y,z) ;

% -------------------------------------------------------------------------
function [inputs2] = getDagNNBatch(imdb, batch, sigma)
% -------------------------------------------------------------------------
patch_size = imdb.patch_size ;
global CurTask;
labels  = zeros(patch_size, patch_size, 1, numel(batch));
for ii = 1:numel(batch)
    labels(:,:,:,ii) = im2single(imread(fullfile(imdb.imdbPath, imdb.filepaths(batch(ii)).name )));
    switch CurTask
        case 'Denoising'
            inputs(:,:,:,ii)  = labels(:,:,:,ii) + single(sigma/255*randn(size(labels(:,:,:,ii))));
        case 'Deblocking'
            str = [ './tmp/' imdb.modelName '_' num2str(sigma) '_' imdb.filepaths(batch(ii)).name(1:end-4) '.jpg'];
            imwrite(labels(:,:,:,ii), str, 'jpg', 'quality', sigma);
            inputs(:,:,:,ii) = im2single(imread(str));
            delete(str);
        case 'SISR'
            inputs(:,:,:,ii) = imresize(imresize(labels(:,:,:,ii), 1/sigma, 'bicubic'), sigma, 'bicubic');
    end
end



        



rng('shuffle');
mode = randperm(8);
inputs = data_augmentation(inputs, mode(1));
labels = data_augmentation(labels, mode(1));


inputs  = gpuArray(inputs)*4-2;
labels= gpuArray(labels)*4-2;


inputs2 = {'input', inputs, 'label', labels} ;



