function output1 = Processing_Im(im, net, gpu, out_idx)
    
    input = im2single(im(:,:,1));
    input = input*4-2;
    if gpu
        input = gpuArray(input);
    end
    net.eval({'input',input}) ;
    %%% output (single)
    output1 = gather(squeeze(gather(net.vars(out_idx).value+2)/4));
    output1 = im2uint8(output1);
end

