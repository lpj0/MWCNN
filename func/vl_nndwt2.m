function Y = vl_nndwt2( X, dzdy, varargin )
% dwt multi_channel 
% only support haart and db2 for 1-level transformation

opts.padding = 0 ;
opts.wavename = 'haart';
opts = vl_argparse(opts, varargin, 'nonrecursive') ;
padding = opts.padding;

if  nargin <= 1 || isempty(dzdy)
    sz = size(X);
    X = X/2;
    if size(X, 3) == 1
        sz(3) = 1;
    end
    if size(X, 4) == 1
        sz(4) = 1;
    end
    
    switch opts.wavename
        case 'pad'
            im_c1 = X(1:2:end, 1:2:end, :, :);
            im_c2 = X(1:2:end, 2:2:end, :, :);
            im_c3 = X(2:2:end, 1:2:end, :, :);
            im_c4 = X(2:2:end, 2:2:end, :, :);
            Y = zeros([sz(1)/2 sz(2)/2 sz(3)*4 sz(4)], 'like', X);
            Y(:,:,1:sz(3),:) = im_c1;
            Y(:,:,sz(3)+1:sz(3)*2,:) = im_c2;
            Y(:,:,sz(3)*2+1:sz(3)*3,:) = im_c3;
            Y(:,:,sz(3)*3+1:sz(3)*4,:) = im_c4;
        case 'haart'
            im_c1 = X(1:2:end, 1:2:end, :, :);
            im_c2 = X(1:2:end, 2:2:end, :, :);
            im_c3 = X(2:2:end, 1:2:end, :, :);
            im_c4 = X(2:2:end, 2:2:end, :, :);
            Y = zeros([sz(1)/2 sz(2)/2 sz(3)*4 sz(4)], 'like', X);
%             Y(:,:,1:sz(3),:)           = bsxfun(@plus, bsxfun(@plus, im_c1, im_c2), bsxfun(@plus, im_c3, im_c4));
%             Y(:,:,sz(3)+1:sz(3)*2,:)   = bsxfun(@minus, bsxfun(@plus, im_c3, im_c4), bsxfun(@plus, im_c1, im_c2));
%             Y(:,:,sz(3)*2+1:sz(3)*3,:) = bsxfun(@minus, bsxfun(@plus, im_c2, im_c4), bsxfun(@plus, im_c1, im_c3));
%             Y(:,:,sz(3)*3+1:end,:)     = bsxfun(@minus, bsxfun(@plus, im_c1, im_c4), bsxfun(@plus, im_c2, im_c3));
            Y(:,:,1:sz(3),:)           = im_c1 + im_c2 + im_c3 + im_c4;
            Y(:,:,sz(3)+1:sz(3)*2,:)   = -im_c1 - im_c2 + im_c3 + im_c4;
            Y(:,:,sz(3)*2+1:sz(3)*3,:) = -im_c1 + im_c2 - im_c3 + im_c4;
            Y(:,:,sz(3)*3+1:end,:)     = im_c1 - im_c2 - im_c3 + im_c4;
         case 'db2'
%           filter_k = zeros(4, 'like', X);
            filter_k = single([0.23325318,-0.0625,-0.0625,0.016746825;0.40400636,-0.10825317,-0.10825317,0.029006351;
                    0.10825317,-0.029006351,0.40400636,-0.10825317;-0.0625,0.016746825,-0.23325318,0.0625;
                    0.40400636,-0.10825317,-0.10825317,0.029006351;0.69975954,-0.1875,-0.1875,0.050240472;
                    0.1875,-0.050240472,0.69975954,-0.1875;-0.10825317,0.029006351,-0.40400636,0.10825317;
                    0.10825317,0.40400636,-0.029006351,-0.10825317;0.1875,0.699759540,-0.050240472,-0.1875;
                    0.050240472,0.1875,0.1875,0.69975954;-0.029006351,-0.10825317,-0.10825317,-0.40400636;
                    -0.0625,-0.23325318,0.016746825,0.0625;-0.10825317,-0.40400636,0.029006351,0.10825317;
                    -0.029006351,-0.10825317,-0.10825317,-0.40400636;0.016746825,0.0625,0.0625,0.23325318]');
            X = padarray(X, [padding(1) padding(1)]);
            if padding(1) ~= padding(2)
                X(end+1,:,:,:) = 0;
                X(:,end+1,:,:) = 0;
            end
            sz(1) = sz(1) + padding(1) + padding(2);
            sz(2) = sz(2) + padding(1) + padding(2);
            if mod(size(X, 1), 2) == 1
                X(end+1,:,:,:) = 0;
                X(:,end+1,:,:) = 0;
            end
            Y = zeros([floor(sz(1)/2)-1 floor(sz(2)/2)-1 sz(3)*4 sz(4)], 'like', X);
            channel_1 = X(1:2:end,1:2:end,:,:);
            channel_2 = X(1:2:end,2:2:end,:,:);
            channel_3 = X(2:2:end,1:2:end,:,:);
            channel_4 = X(2:2:end,2:2:end,:,:);
%             Y(:,:,1:sz(3),:) = Y(:,:,1:sz(3),:) + filter_k(1,1)*channel_1(1:end-1,1:end-1,:,:)+...
%                 filter_k(1,2)*channel_2(1:end-1,1:end-1,:,:)+filter_k(1,3)*channel_3(1:end-1,2:end,:,:)...
%                 + filter_k(1,4)*channel_4(1:end-1,2:end,:,:)+ filter_k(1,5)*channel_1(1:end-1,1:end-1,:,:)...
%                 + filter_k(1,6)*channel_2(1:end-1,1:end-1,:,:)+ filter_k(1,7)*channel_3(1:end-1,2:end,:,:)...
%                 + filter_k(1,8)*channel_4(1:end-1,2:end,:,:)+ filter_k(1,9)*channel_1(2:end,1:end-1,:,:)...
%                 + filter_k(1,10)*channel_2(2:end,1:end-1,:,:)+ filter_k(1,11)*channel_3(2:end,2:end,:,:)...
%                 + filter_k(1,12)*channel_4(2:end,2:end,:,:)+ filter_k(1,13)*channel_1(2:end,1:end-1,:,:)...
%                 + filter_k(1,14)*channel_2(2:end,1:end-1,:,:)+ filter_k(1,15)*channel_3(2:end,2:end,:,:)...
%                 + filter_k(1,16)*channel_4(2:end,2:end,:,:); 
%             Y(:,:,sz(3)+1:sz(3)*2,:) = Y(:,:,sz(3)+1:sz(3)*2,:) + filter_k(2,1)*channel_1(1:end-1,1:end-1,:,:)+...
%                 filter_k(1,2)*channel_2(1:end-1,1:end-1,:,:)+filter_k(2,3)*channel_3(1:end-1,2:end,:,:)...
%                 + filter_k(2,4)*channel_4(1:end-1,2:end,:,:)+ filter_k(2,5)*channel_1(1:end-1,1:end-1,:,:)...
%                 + filter_k(2,6)*channel_2(1:end-1,1:end-1,:,:)+ filter_k(2,7)*channel_3(1:end-1,2:end,:,:)...
%                 + filter_k(2,8)*channel_4(1:end-1,2:end,:,:)+ filter_k(2,9)*channel_1(2:end,1:end-1,:,:)...
%                 + filter_k(2,10)*channel_2(2:end,1:end-1,:,:)+ filter_k(2,11)*channel_3(2:end,2:end,:,:)...
%                 + filter_k(2,12)*channel_4(2:end,2:end,:,:)+ filter_k(2,13)*channel_1(2:end,1:end-1,:,:)...
%                 + filter_k(2,14)*channel_2(2:end,1:end-1,:,:)+ filter_k(2,15)*channel_3(2:end,2:end,:,:)...
%                 + filter_k(2,16)*channel_4(2:end,2:end,:,:);   
%             Y(:,:,sz(3)*2+1:sz(3)*3,:) = Y(:,:,sz(3)*2+1:sz(3)*3,:) + filter_k(3,1)*channel_1(1:end-1,1:end-1,:,:)+...
%                 filter_k(1,2)*channel_2(1:end-1,1:end-1,:,:)+filter_k(3,3)*channel_3(1:end-1,2:end,:,:)...
%                 + filter_k(3,4)*channel_4(1:end-1,2:end,:,:)+ filter_k(3,5)*channel_1(1:end-1,1:end-1,:,:)...
%                 + filter_k(3,6)*channel_2(1:end-1,1:end-1,:,:)+ filter_k(3,7)*channel_3(1:end-1,2:end,:,:)...
%                 + filter_k(3,8)*channel_4(1:end-1,2:end,:,:)+ filter_k(3,9)*channel_1(2:end,1:end-1,:,:)...
%                 + filter_k(3,10)*channel_2(2:end,1:end-1,:,:)+ filter_k(3,11)*channel_3(2:end,2:end,:,:)...
%                 + filter_k(3,12)*channel_4(2:end,2:end,:,:)+ filter_k(3,13)*channel_1(2:end,1:end-1,:,:)...
%                 + filter_k(3,14)*channel_2(2:end,1:end-1,:,:)+ filter_k(3,15)*channel_3(2:end,2:end,:,:)...
%                 + filter_k(3,16)*channel_4(2:end,2:end,:,:);  
%             Y(:,:,sz(3)*3+1:end,:) = Y(:,:,sz(3)*3+1:end,:) + filter_k(4,1)*channel_1(1:end-1,1:end-1,:,:)+...
%                 filter_k(1,2)*channel_2(1:end-1,1:end-1,:,:)+filter_k(4,3)*channel_3(1:end-1,2:end,:,:)...
%                 + filter_k(4,4)*channel_4(1:end-1,2:end,:,:)+ filter_k(4,5)*channel_1(1:end-1,1:end-1,:,:)...
%                 + filter_k(4,6)*channel_2(1:end-1,1:end-1,:,:)+ filter_k(4,7)*channel_3(1:end-1,2:end,:,:)...
%                 + filter_k(4,8)*channel_4(1:end-1,2:end,:,:)+ filter_k(4,9)*channel_1(2:end,1:end-1,:,:)...
%                 + filter_k(4,10)*channel_2(2:end,1:end-1,:,:)+ filter_k(4,11)*channel_3(2:end,2:end,:,:)...
%                 + filter_k(4,12)*channel_4(2:end,2:end,:,:)+ filter_k(4,13)*channel_1(2:end,1:end-1,:,:)...
%                 + filter_k(4,14)*channel_2(2:end,1:end-1,:,:)+ filter_k(4,15)*channel_3(2:end,2:end,:,:)...
%                 + filter_k(4,16)*channel_4(2:end,2:end,:,:);  
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
                Y(:,:,1:sz(3),:) =  Y(:,:,1:sz(3),:) + filter_k(1,ii)*channel_x; 
                Y(:,:,sz(3)+1:sz(3)*2,:) = Y(:,:,sz(3)+1:sz(3)*2,:) + filter_k(2,ii)*channel_x; 
                Y(:,:,sz(3)*2+1:sz(3)*3,:) = Y(:,:,sz(3)*2+1:sz(3)*3,:) + filter_k(3,ii)*channel_x; 
                Y(:,:,sz(3)*3+1:end,:) = Y(:,:,sz(3)*3+1:end,:) + filter_k(4,ii)*channel_x; 
%                 Y(:,:,1:sz(3),:) = bsxfun(@plus, Y(:,:,1:sz(3),:), bsxfun(@times, filter_k(1,ii), channel_x)); 
%                 Y(:,:,sz(3)+1:sz(3)*2,:) = bsxfun(@plus, Y(:,:,sz(3)+1:sz(3)*2,:), bsxfun(@times, filter_k(2,ii), channel_x));  
%                 Y(:,:,sz(3)*2+1:sz(3)*3,:) = bsxfun(@plus, Y(:,:,sz(3)*2+1:sz(3)*3,:), bsxfun(@times, filter_k(3,ii), channel_x)); 
%                 Y(:,:,sz(3)*3+1:end,:) = bsxfun(@plus, Y(:,:,sz(3)*3+1:end,:), bsxfun(@times, filter_k(4,ii), channel_x));  
            end
    end
else
    sz = size(dzdy);
    if size(X, 4) == 1
        sz(4) = 1;
    end
    switch opts.wavename
        case 'pad'
            Y = zeros([sz(1)*2 sz(2)*2 sz(3)/4 sz(4)],'like',dzdy);
            Y(1:2:end, 1:2:end, : , :) = dzdy(:,:,1:sz(3)/4,:);
            Y(1:2:end, 2:2:end, : , :) = dzdy(:,:,sz(3)/4+1:sz(3)/2,:);
            Y(2:2:end, 1:2:end, : , :) = dzdy(:,:,sz(3)/2+1:3*sz(3)/4,:);
            Y(2:2:end, 2:2:end, : , :) = dzdy(:,:,sz(3)/4*3+1:end,:);
        case 'haart'
            dzdy = dzdy/2;
            Y = zeros([sz(1)*2 sz(2)*2 sz(3)/4 sz(4)],'like',dzdy);
%             Y(1:2:end, 1:2:end, : , :) = bsxfun(@plus, bsxfun(@minus, dzdy(:,:,1:sz(3)/4,:), dzdy(:,:,sz(3)/4+1:sz(3)/2,:)),...
%                                          bsxfun(@minus, dzdy(:,:,sz(3)/4*3+1:end,:), dzdy(:,:,sz(3)/2+1:3*sz(3)/4,:)));
%                                      
%             Y(1:2:end, 2:2:end, : , :) = bsxfun(@plus, bsxfun(@minus, dzdy(:,:,1:sz(3)/4,:), dzdy(:,:,sz(3)/4+1:sz(3)/2,:)),...
%                                          bsxfun(@minus, dzdy(:,:,sz(3)/2+1:3*sz(3)/4,:), dzdy(:,:,sz(3)/4*3+1:end,:)));
%                                      
%             Y(2:2:end, 1:2:end, : , :) = bsxfun(@minus, bsxfun(@plus, dzdy(:,:,1:sz(3)/4,:), dzdy(:,:,sz(3)/4+1:sz(3)/2,:)),...
%                                          bsxfun(@plus, dzdy(:,:,sz(3)/2+1:3*sz(3)/4,:), dzdy(:,:,sz(3)/4*3+1:end,:)));
%                                      
%             Y(2:2:end, 2:2:end, : , :) = bsxfun(@plus, bsxfun(@plus, dzdy(:,:,1:sz(3)/4,:), dzdy(:,:,sz(3)/4+1:sz(3)/2,:)),...
%                                          bsxfun(@plus, dzdy(:,:,sz(3)/2+1:3*sz(3)/4,:), dzdy(:,:,sz(3)/4*3+1:end,:)));
            Y(1:2:end, 1:2:end, : , :) = dzdy(:,:,1:sz(3)/4,:) - dzdy(:,:,sz(3)/4+1:sz(3)/2,:) - ... 
                                         dzdy(:,:,sz(3)/2+1:3*sz(3)/4,:) + dzdy(:,:,sz(3)/4*3+1:end,:);
            Y(1:2:end, 2:2:end, : , :) = dzdy(:,:,1:sz(3)/4,:) - dzdy(:,:,sz(3)/4+1:sz(3)/2,:) + ...
                                         dzdy(:,:,sz(3)/2+1:3*sz(3)/4,:) - dzdy(:,:,sz(3)/4*3+1:end,:);
            Y(2:2:end, 1:2:end, : , :) = dzdy(:,:,1:sz(3)/4,:) + dzdy(:,:,sz(3)/4+1:sz(3)/2,:) - ...
                                         dzdy(:,:,sz(3)/2+1:3*sz(3)/4,:) - dzdy(:,:,sz(3)/4*3+1:end,:);
            Y(2:2:end, 2:2:end, : , :) = dzdy(:,:,1:sz(3)/4,:) + dzdy(:,:,sz(3)/4+1:sz(3)/2,:) + ...
                                         dzdy(:,:,sz(3)/2+1:3*sz(3)/4,:) + dzdy(:,:,sz(3)/4*3+1:end,:);
        case 'db2'
            filter_k = single([0.23325318,-0.0625,-0.0625,0.016746825;0.40400636,-0.10825317,-0.10825317,0.029006351;
                0.10825317,-0.029006351,0.40400636,-0.10825317;-0.0625,0.016746825,-0.23325318,0.0625;
                0.40400636,-0.10825317,-0.10825317,0.029006351;0.69975954,-0.1875,-0.1875,0.050240472;
                0.1875,-0.050240472,0.69975954,-0.1875;-0.10825317,0.029006351,-0.40400636,0.10825317;
                0.10825317,0.40400636,-0.029006351,-0.10825317;0.1875,0.699759540,-0.050240472,-0.1875;
                0.050240472,0.1875,0.1875,0.69975954;-0.029006351,-0.10825317,-0.10825317,-0.40400636;
                -0.0625,-0.23325318,0.016746825,0.0625;-0.10825317,-0.40400636,0.029006351,0.10825317;
                -0.029006351,-0.10825317,-0.10825317,-0.40400636;0.016746825,0.0625,0.0625,0.23325318]');
            Y = zeros([sz(1)*2+2 sz(2)*2+2 sz(3)/4 sz(4)], 'like', dzdy);
            channel_1 = dzdy(:,:,1:sz(3)/4,:);
            channel_2 = dzdy(:,:,sz(3)/4+1:sz(3)/2,:);
            channel_3 = dzdy(:,:,sz(3)/2+1:sz(3)*3/4,:);
            channel_4 = dzdy(:,:,sz(3)*3/4+1:end,:);
            for  ii = 1:16
                vs = mod(ii-1, 4)+1; 
                hs = floor((ii-1)/4)+1;
                if  hs <=2 && vs <=2 
                     Y(hs:2:end-2,vs:2:end-2,:,:) =  Y(hs:2:end-2,vs:2:end-2,:,:) +... 
                         filter_k(1,ii)*channel_1 + filter_k(2,ii)*channel_2+...
                         filter_k(3,ii)*channel_3 + filter_k(4,ii)*channel_4; 
%                     Y(hs:2:end-2,vs:2:end-2,:,:) = bsxfun(@plus, Y(hs:2:end-2,vs:2:end-2,:,:),... 
%                          bsxfun(@plus, bsxfun(@plus, bsxfun(@times, filter_k(1,ii), channel_1), bsxfun(@times, filter_k(2,ii), channel_2)), ...
%                          bsxfun(@plus, bsxfun(@times, filter_k(3,ii), channel_3), bsxfun(@times, filter_k(4,ii), channel_4)))); 
                end
                if  hs <=2 && vs >2
                    Y(hs:2:end-2,vs:2:end,:,:) = Y(hs:2:end-2,vs:2:end,:,:) + ... 
                         filter_k(1,ii)*channel_1 + filter_k(2,ii)*channel_2 + ...
                         filter_k(3,ii)*channel_3 + filter_k(4,ii)*channel_4; 
%                     Y(hs:2:end-2,vs:2:end,:,:) = bsxfun(@plus, Y(hs:2:end-2,vs:2:end,:,:),... 
%                          bsxfun(@plus, bsxfun(@plus, bsxfun(@times, filter_k(1,ii), channel_1), bsxfun(@times, filter_k(2,ii), channel_2)), ...
%                          bsxfun(@plus, bsxfun(@times, filter_k(3,ii), channel_3), bsxfun(@times, filter_k(4,ii), channel_4)))); 
                end
                if  hs > 2 && vs <=2
                      Y(hs:2:end,vs:2:end-2,:,:) = Y(hs:2:end,vs:2:end-2,:,:) + ... 
                         filter_k(1,ii)*channel_1 + filter_k(2,ii)*channel_2 + ...
                         filter_k(3,ii)*channel_3 + filter_k(4,ii)*channel_4;
%                     Y(hs:2:end,vs:2:end-2,:,:) = bsxfun(@plus, Y(hs:2:end,vs:2:end-2,:,:),... 
%                          bsxfun(@plus, bsxfun(@plus, bsxfun(@times, filter_k(1,ii), channel_1), bsxfun(@times, filter_k(2,ii), channel_2)), ...
%                          bsxfun(@plus, bsxfun(@times, filter_k(3,ii), channel_3), bsxfun(@times, filter_k(4,ii), channel_4)))); 
                end
                if  hs > 2 && vs > 2
                    Y(hs:2:end,vs:2:end,:,:) = Y(hs:2:end,vs:2:end,:,:) + ... 
                         filter_k(1,ii)*channel_1 + filter_k(2,ii)*channel_2 + ...
                         filter_k(3,ii)*channel_3 + filter_k(4,ii)*channel_4;
%                     Y(hs:2:end,vs:2:end,:,:) = bsxfun(@plus, Y(hs:2:end,vs:2:end,:,:),... 
%                          bsxfun(@plus, bsxfun(@plus, bsxfun(@times, filter_k(1,ii), channel_1), bsxfun(@times, filter_k(2,ii), channel_2)), ...
%                          bsxfun(@plus, bsxfun(@times, filter_k(3,ii), channel_3), bsxfun(@times, filter_k(4,ii), channel_4)))); 
                end
            end
            Y = Y(1+padding(1):end-padding(2),1+padding(1):end-padding(2),:,:);
    end
end

