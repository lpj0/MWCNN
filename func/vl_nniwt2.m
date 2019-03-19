function Y = vl_nniwt2( X, dzdy, varargin )
% iwt multi_channel 
% only support haart and db2 for 1-level transformation

opts.padding = 0 ;
opts.wavename = 'haart';
opts = vl_argparse(opts, varargin, 'nonrecursive') ;
padding = opts.padding;
if  nargin <= 1 || isempty(dzdy)
    sz = size(X);
    if size(X, 4) == 1
        sz(4) = 1;
    end
    switch opts.wavename
        case 'pad'
            Y = zeros([sz(1)*2 sz(2)*2 sz(3)/4 sz(4)],'like',X);
            Y(1:2:end, 1:2:end, : , :) = X(:,:,1:sz(3)/4,:);
            Y(1:2:end, 2:2:end, : , :) = X(:,:,sz(3)/4+1:sz(3)/2,:);
            Y(2:2:end, 1:2:end, : , :) = X(:,:,sz(3)/2+1:3*sz(3)/4,:);
            Y(2:2:end, 2:2:end, : , :) = X(:,:,sz(3)/4*3+1:end,:);
        case 'haart'
            X = X/2;
            Y = zeros([sz(1)*2 sz(2)*2 sz(3)/4 sz(4)],'like',X);
%             Y(1:2:end, 1:2:end, : , :) = bsxfun(@plus, bsxfun(@minus, X(:,:,1:sz(3)/4,:), X(:,:,sz(3)/4+1:sz(3)/2,:)),...
%                                          bsxfun(@minus, X(:,:,sz(3)/4*3+1:end,:), X(:,:,sz(3)/2+1:3*sz(3)/4,:)));
%                                      
%             Y(1:2:end, 2:2:end, : , :) = bsxfun(@plus, bsxfun(@minus, X(:,:,1:sz(3)/4,:), X(:,:,sz(3)/4+1:sz(3)/2,:)),...
%                                          bsxfun(@minus, X(:,:,sz(3)/2+1:3*sz(3)/4,:), X(:,:,sz(3)/4*3+1:end,:)));
%                                      
%             Y(2:2:end, 1:2:end, : , :) = bsxfun(@minus, bsxfun(@plus, X(:,:,1:sz(3)/4,:), X(:,:,sz(3)/4+1:sz(3)/2,:)),...
%                                          bsxfun(@plus, X(:,:,sz(3)/2+1:3*sz(3)/4,:), X(:,:,sz(3)/4*3+1:end,:)));
%                                      
%             Y(2:2:end, 2:2:end, : , :) = bsxfun(@plus, bsxfun(@plus, X(:,:,1:sz(3)/4,:), X(:,:,sz(3)/4+1:sz(3)/2,:)),...
%                                          bsxfun(@plus, X(:,:,sz(3)/2+1:3*sz(3)/4,:), X(:,:,sz(3)/4*3+1:end,:)));
            Y(1:2:end, 1:2:end, : , :) = X(:,:,1:sz(3)/4,:) - X(:,:,sz(3)/4+1:sz(3)/2,:) - X(:,:,sz(3)/2+1:3*sz(3)/4,:) + X(:,:,sz(3)/4*3+1:end,:);
            Y(1:2:end, 2:2:end, : , :) = X(:,:,1:sz(3)/4,:) - X(:,:,sz(3)/4+1:sz(3)/2,:) + X(:,:,sz(3)/2+1:3*sz(3)/4,:) - X(:,:,sz(3)/4*3+1:end,:);
            Y(2:2:end, 1:2:end, : , :) = X(:,:,1:sz(3)/4,:) + X(:,:,sz(3)/4+1:sz(3)/2,:) - X(:,:,sz(3)/2+1:3*sz(3)/4,:) - X(:,:,sz(3)/4*3+1:end,:);
            Y(2:2:end, 2:2:end, : , :) = X(:,:,1:sz(3)/4,:) + X(:,:,sz(3)/4+1:sz(3)/2,:) + X(:,:,sz(3)/2+1:3*sz(3)/4,:) + X(:,:,sz(3)/4*3+1:end,:);
        case 'db2'
            filter_k = single([0.23325318,-0.0625,-0.0625,0.016746825;0.40400636,-0.10825317,-0.10825317,0.029006351;
                0.10825317,-0.029006351,0.40400636,-0.10825317;-0.0625,0.016746825,-0.23325318,0.0625;
                0.40400636,-0.10825317,-0.10825317,0.029006351;0.69975954,-0.1875,-0.1875,0.050240472;
                0.1875,-0.050240472,0.69975954,-0.1875;-0.10825317,0.029006351,-0.40400636,0.10825317;
                0.10825317,0.40400636,-0.029006351,-0.10825317;0.1875,0.699759540,-0.050240472,-0.1875;
                0.050240472,0.1875,0.1875,0.69975954;-0.029006351,-0.10825317,-0.10825317,-0.40400636;
                -0.0625,-0.23325318,0.016746825,0.0625;-0.10825317,-0.40400636,0.029006351,0.10825317;
                -0.029006351,-0.10825317,-0.10825317,-0.40400636;0.016746825,0.0625,0.0625,0.23325318]');
            Y = zeros([sz(1)*2+2 sz(2)*2+2 sz(3)/4 sz(4)], 'like', X);
            channel_1 = X(:,:,1:sz(3)/4,:);
            channel_2 = X(:,:,sz(3)/4+1:sz(3)/2,:);
            channel_3 = X(:,:,sz(3)/2+1:sz(3)*3/4,:);
            channel_4 = X(:,:,sz(3)*3/4+1:end,:);
            for  ii = 1:16
                vs = mod(ii-1, 4)+1; 
                hs = floor((ii-1)/4)+1;
                if  hs <=2 && vs <=2 
                     Y(hs:2:end-2,vs:2:end-2,:,:) = Y(hs:2:end-2,vs:2:end-2,:,:) + ...
                         filter_k(1,ii)*channel_1  + filter_k(2,ii)*channel_2 + ...
                         filter_k(3,ii)*channel_3  + filter_k(4,ii)*channel_4; 
                end
                if  hs <=2 && vs >2
                     Y(hs:2:end-2,vs:2:end,:,:)   = Y(hs:2:end-2,vs:2:end,:,:) + ...
                         filter_k(1,ii) + channel_1 + filter_k(2,ii) * channel_2 + ...
                         filter_k(3,ii) + channel_3 + filter_k(4,ii) * channel_4;  
                end
                if  hs > 2 && vs <=2
                    Y(hs:2:end,vs:2:end-2,:,:)    = Y(hs:2:end,vs:2:end-2,:,:) + ... 
                         filter_k(1,ii)*channel_1 + filter_k(2,ii)*channel_2 + ...
                         filter_k(3,ii)*channel_3 + filter_k(4,ii)*channel_4;  
                end
                if  hs > 2 && vs > 2
                    Y(hs:2:end,vs:2:end,:,:)      = Y(hs:2:end,vs:2:end,:,:) + ...
                         filter_k(1,ii)*channel_1 + filter_k(2,ii)*channel_2 + ...
                         filter_k(3,ii)*channel_3 + filter_k(4,ii)*channel_4; 
                end
            end
%             for  ii = 1:16
%                 vs = mod(ii-1, 4)+1; 
%                 hs = floor((ii-1)/4)+1;
%                 if  hs <=2 && vs <=2 
%                      Y(hs:2:end-2,vs:2:end-2,:,:) = bsxfun(@plus, Y(hs:2:end-2,vs:2:end-2,:,:) ,...
%                          bsxfun(@plus, bsxfun(@plus, bsxfun(@times, filter_k(1,ii), channel_1), bsxfun(@times, filter_k(2,ii), channel_2)), ...
%                          bsxfun(@plus, bsxfun(@times, filter_k(3,ii), channel_3), bsxfun(@times,filter_k(4,ii), channel_4)))); 
%                 end
%                 if  hs <=2 && vs >2
%                      Y(hs:2:end-2,vs:2:end,:,:)   = bsxfun(@plus, Y(hs:2:end-2,vs:2:end,:,:),...
%                          bsxfun(@plus, bsxfun(@plus, bsxfun(@times, filter_k(1,ii), channel_1), bsxfun(@times, filter_k(2,ii), channel_2)), ...
%                          bsxfun(@plus, bsxfun(@times, filter_k(3,ii), channel_3), bsxfun(@times,filter_k(4,ii), channel_4))));  
%                 end
%                 if  hs > 2 && vs <=2
%                     Y(hs:2:end,vs:2:end-2,:,:)    = bsxfun(@plus, Y(hs:2:end,vs:2:end-2,:,:),... 
%                          bsxfun(@plus, bsxfun(@plus, bsxfun(@times, filter_k(1,ii), channel_1), bsxfun(@times, filter_k(2,ii), channel_2)), ...
%                          bsxfun(@plus, bsxfun(@times, filter_k(3,ii), channel_3), bsxfun(@times,filter_k(4,ii), channel_4))));  
%                 end
%                 if  hs > 2 && vs > 2
%                     Y(hs:2:end,vs:2:end,:,:)      = bsxfun(@plus, Y(hs:2:end,vs:2:end,:,:),...
%                          bsxfun(@plus, bsxfun(@plus, bsxfun(@times, filter_k(1,ii), channel_1), bsxfun(@times, filter_k(2,ii), channel_2)), ...
%                          bsxfun(@plus, bsxfun(@times, filter_k(3,ii), channel_3), bsxfun(@times,filter_k(4,ii), channel_4)))); 
%                 end
%             end
            Y = Y(1+padding(1):end-padding(2),1+padding(1):end-padding(2),:,:);
    end

else
    sz = size(dzdy);
    
    if size(dzdy, 3) == 1
        sz(3) = 1;
    end
    if size(dzdy, 4) == 1
        sz(4) = 1;
    end
    switch opts.wavename
        case 'pad'
            im_c1 = dzdy(1:2:end, 1:2:end, :, :);
            im_c2 = dzdy(1:2:end, 2:2:end, :, :);
            im_c3 = dzdy(2:2:end, 1:2:end, :, :);
            im_c4 = dzdy(2:2:end, 2:2:end, :, :);
            Y = zeros([sz(1)/2 sz(2)/2 sz(3)*4 sz(4)], 'like', dzdy);
            Y(:,:,1:sz(3),:) = im_c1;
            Y(:,:,sz(3)+1:sz(3)*2,:) = im_c2 ;
            Y(:,:,sz(3)*2+1:sz(3)*3,:) = im_c3;
            Y(:,:,sz(3)*3+1:sz(3)*4,:) = im_c4;
        case 'haart'
            dzdy = dzdy/2;
            im_c1 = dzdy(1:2:end, 1:2:end, :, :);
            im_c2 = dzdy(1:2:end, 2:2:end, :, :);
            im_c3 = dzdy(2:2:end, 1:2:end, :, :);
            im_c4 = dzdy(2:2:end, 2:2:end, :, :);
            Y = zeros([sz(1)/2 sz(2)/2 sz(3)*4 sz(4)], 'like', dzdy);
%             Y(:,:,1:sz(3),:)           = bsxfun(@plus, bsxfun(@plus, im_c1, im_c2), bsxfun(@plus, im_c3, im_c4));
%             Y(:,:,sz(3)+1:sz(3)*2,:)   = bsxfun(@minus, bsxfun(@plus, im_c3, im_c4), bsxfun(@plus, im_c1, im_c2));
%             Y(:,:,sz(3)*2+1:sz(3)*3,:) = bsxfun(@minus, bsxfun(@plus, im_c2, im_c4), bsxfun(@plus, im_c1, im_c3));
%             Y(:,:,sz(3)*3+1:end,:)     = bsxfun(@minus, bsxfun(@plus, im_c1, im_c4), bsxfun(@plus, im_c2, im_c3));
            Y(:,:,1:sz(3),:) = im_c1 + im_c2 + im_c3 + im_c4;
            Y(:,:,sz(3)+1:sz(3)*2,:) = -im_c1 - im_c2 + im_c3 + im_c4;
            Y(:,:,sz(3)*2+1:sz(3)*3,:) = -im_c1 + im_c2 - im_c3 + im_c4;
            Y(:,:,sz(3)*3+1:sz(3)*4,:) = im_c1 - im_c2 - im_c3 + im_c4;
         case 'db2'
             filter_k = gpuArray(single([0.23325318,-0.0625,-0.0625,0.016746825;0.40400636,-0.10825317,-0.10825317,0.029006351;
                0.10825317,-0.029006351,0.40400636,-0.10825317;-0.0625,0.016746825,-0.23325318,0.0625;
                0.40400636,-0.10825317,-0.10825317,0.029006351;0.69975954,-0.1875,-0.1875,0.050240472;
                0.1875,-0.050240472,0.69975954,-0.1875;-0.10825317,0.029006351,-0.40400636,0.10825317;
                0.10825317,0.40400636,-0.029006351,-0.10825317;0.1875,0.699759540,-0.050240472,-0.1875;
                0.050240472,0.1875,0.1875,0.69975954;-0.029006351,-0.10825317,-0.10825317,-0.40400636;
                -0.0625,-0.23325318,0.016746825,0.0625;-0.10825317,-0.40400636,0.029006351,0.10825317;
                -0.029006351,-0.10825317,-0.10825317,-0.40400636;0.016746825,0.0625,0.0625,0.23325318]'));
            dzdy = padarray(dzdy, [padding(1) padding(1)]);
            if padding(1) ~= padding(2)
                dzdy(end+1,:,:,:) = 0;
                dzdy(:,end+1,:,:) = 0;
            end
            sz(1) = sz(1) + padding(1) + padding(2);
            sz(2) = sz(2) + padding(1) + padding(2);
            Y = zeros([floor(sz(1)/2)-1 floor(sz(2)/2)-1 sz(3)*4 sz(4)], 'like', dzdy);
            channel_1 = dzdy(1:2:end,1:2:end,:,:);
            channel_2 = dzdy(1:2:end,2:2:end,:,:);
            channel_3 = dzdy(2:2:end,1:2:end,:,:);
            channel_4 = dzdy(2:2:end,2:2:end,:,:);

            for ii = 1:16
                vs = mod(ii-1, 4)+1; 
                hs = floor((ii-1)/4)+1;
                if  (vs == 1 || vs == 3) && (hs == 1 || hs == 3)
                    channel_x = channel_1(ceil(hs/2):end+ceil(hs/2)-2 , ceil(vs/2):end+ceil(vs/2)-2,:,:);
                end
                if  (hs == 1 || hs == 3) && (vs == 2 || vs == 4)
                    channel_x = channel_2(ceil(hs/2):end+ceil(hs/2)-2 , ceil(vs/2):end+ceil(vs/2)-2,:,:);
                end
                if  (hs == 2 || hs == 4) && (vs == 1 || vs == 3)
                    channel_x = channel_3(ceil(hs/2):end+ceil(hs/2)-2 , ceil(vs/2):end+ceil(vs/2)-2,:,:);
                end
                if  (hs == 2 || hs == 4) && (vs == 2 || vs == 4)
                    channel_x = channel_4(ceil(hs/2):end+ceil(hs/2)-2 , ceil(vs/2):end+ceil(vs/2)-2,:,:);
                end
                Y(:,:,1:sz(3),:) = Y(:,:,1:sz(3),:) + filter_k(1,ii)*channel_x; 
                Y(:,:,sz(3)+1:sz(3)*2,:) =  Y(:,:,sz(3)+1:sz(3)*2,:) + filter_k(2,ii)*channel_x;  
                Y(:,:,sz(3)*2+1:sz(3)*3,:) = Y(:,:,sz(3)*2+1:sz(3)*3,:) + filter_k(3,ii)*channel_x; 
                Y(:,:,sz(3)*3+1:end,:) = Y(:,:,sz(3)*3+1:end,:) + filter_k(4,ii)*channel_x; 
%                 Y(:,:,1:sz(3),:) = bsxfun(@plus, Y(:,:,1:sz(3),:), bsxfun(@times, filter_k(1,ii), channel_x)); 
%                 Y(:,:,sz(3)+1:sz(3)*2,:) = bsxfun(@plus, Y(:,:,sz(3)+1:sz(3)*2,:), bsxfun(@times, filter_k(2,ii), channel_x));  
%                 Y(:,:,sz(3)*2+1:sz(3)*3,:) = bsxfun(@plus, Y(:,:,sz(3)*2+1:sz(3)*3,:), bsxfun(@times, filter_k(3,ii), channel_x)); 
%                 Y(:,:,sz(3)*3+1:end,:) = bsxfun(@plus, Y(:,:,sz(3)*3+1:end,:), bsxfun(@times, filter_k(4,ii), channel_x));  
            end
     end
end

