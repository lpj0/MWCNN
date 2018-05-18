classdef IWT2HD < dagnn.ElementWise
    

  properties (Transient)
      padding = 0
      wavename = 'haart'
      opts = {}
    numInputs
  end

  methods
    function outputs = forward(obj, inputs, params)
      obj.numInputs = numel(inputs) ;
      outputs{1} = vl_nniwt2(inputs{1}, [], ...
          'wavename', obj.wavename, 'padding', obj.padding, obj.opts{:}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} =  vl_nniwt2(inputs{1},  derOutputs{1}, ...
          'wavename', obj.wavename, 'padding', obj.padding, obj.opts{:}) ; 
      derParams = {0} ;
    end
    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = {} ;
    end

    function rfs = getReceptiveFields(obj)
        rfs = [] ;
    end

    function obj = IWT2HD(varargin)
      obj.load(varargin) ;
    end
  end
end
