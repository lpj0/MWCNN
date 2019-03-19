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
            X = X/2;
            Y = zeros([sz(1)*2 sz(2)*2 sz(3)/4 sz(4)],'like',X);
            Y(1:2:end, 1:2:end, : , :) = X(:,:,1:sz(3)/4,:) - X(:,:,sz(3)/4+1:sz(3)/2,:) - X(:,:,sz(3)/2+1:3*sz(3)/4,:) + X(:,:,sz(3)/4*3+1:end,:);
            Y(1:2:end, 2:2:end, : , :) = X(:,:,1:sz(3)/4,:) - X(:,:,sz(3)/4+1:sz(3)/2,:) + X(:,:,sz(3)/2+1:3*sz(3)/4,:) - X(:,:,sz(3)/4*3+1:end,:);
            Y(2:2:end, 1:2:end, : , :) = X(:,:,1:sz(3)/4,:) + X(:,:,sz(3)/4+1:sz(3)/2,:) - X(:,:,sz(3)/2+1:3*sz(3)/4,:) - X(:,:,sz(3)/4*3+1:end,:);
            Y(2:2:end, 2:2:end, : , :) = X(:,:,1:sz(3)/4,:) + X(:,:,sz(3)/4+1:sz(3)/2,:) + X(:,:,sz(3)/2+1:3*sz(3)/4,:) + X(:,:,sz(3)/4*3+1:end,:);

else
    sz = size(dzdy);
    
    if size(dzdy, 3) == 1
        sz(3) = 1;
    end
    if size(dzdy, 4) == 1
        sz(4) = 1;
    end

            dzdy = dzdy/2;
            im_c1 = dzdy(1:2:end, 1:2:end, :, :);
            im_c2 = dzdy(1:2:end, 2:2:end, :, :);
            im_c3 = dzdy(2:2:end, 1:2:end, :, :);
            im_c4 = dzdy(2:2:end, 2:2:end, :, :);
            Y = zeros([sz(1)/2 sz(2)/2 sz(3)*4 sz(4)], 'like', dzdy);
            Y(:,:,1:sz(3),:) = im_c1 + im_c2 + im_c3 + im_c4;
            Y(:,:,sz(3)+1:sz(3)*2,:) = -im_c1 - im_c2 + im_c3 + im_c4;
            Y(:,:,sz(3)*2+1:sz(3)*3,:) = -im_c1 + im_c2 - im_c3 + im_c4;
            Y(:,:,sz(3)*3+1:sz(3)*4,:) = im_c1 - im_c2 - im_c3 + im_c4;
end

