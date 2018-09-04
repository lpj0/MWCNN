
function net = net_wavelet_haart_24()
%D-D-U-U  
% global scale;
feature_map_size = 160;
feature_map_size1 = 256;
feature_map_size2 = 256;
% global feature_map_size3;
%% In writing
% load model_25_29bb-epoch-43.mat
% net = dagnn.DagNN.loadobj(net) ;
%% default setting

h_default            = 3;
w_default            = 3;

%convOpts = {'CudnnWorkspaceLimit', 8 * 1024^3} ;

% Define network
net.layers = {} ;

net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

% net = add_LIEVEN_Haar(net, 'Haart_LIEVE01', {'input','LIEVE00_t00'}, 'Haart_LIEVE01f', [2 2 1 scale^2], scale);

l1 = 1; 
ll1 = 0;

net.addLayer(['Haart_LIEVE'  num2str(l1, '%02d')], dagnn.DWT2HD(), 'input', ['LIEVE' num2str(l1, '%02d') '_t'], {});

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d')], {['LIEVE' num2str(l1, '%02d') '_t'], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll1, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'b']}, [h_default w_default 4 feature_map_size], 1);

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll1, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll1, '%02d')], {});

ll1 = ll1 + 1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll1-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll1, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'b']}, [h_default w_default feature_map_size feature_map_size], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll1, '%02d')]}, feature_map_size);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll1, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll1, '%02d')], {});


ll1 = ll1 + 1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll1-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll1, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'b']}, [h_default w_default feature_map_size feature_map_size], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll1, '%02d')]}, feature_map_size);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll1, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll1, '%02d')], {});

ll1 = ll1 + 1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll1-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll1, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'b']}, [h_default w_default feature_map_size feature_map_size], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll1, '%02d')]}, feature_map_size);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll1, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll1, '%02d')], {});

l1 = l1 + 1; 

% net = add_LIEVEN_Haar(net, 'Haart_LIEVE02', {'LIEVE01_r01','LIEVE02_sx'}, 'Haart_LIEVE02f', [2 2 1 feature_map_size*scale^2], scale);
net.addLayer(['Haart_LIEVE'  num2str(l1, '%02d')], dagnn.DWT2HD(),['LIEVE' num2str(l1-1, '%02d') '_r' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_haartx'], {});

ll2 = 0;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d')], {['LIEVE' num2str(l1, '%02d') '_haartx'], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'b']}, [h_default w_default feature_map_size*4 feature_map_size1], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll2, '%02d')]}, feature_map_size1);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll2, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll2, '%02d')], {});

ll2 = ll2+1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll2-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'b']}, [h_default w_default feature_map_size1 feature_map_size1], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll2, '%02d')]}, feature_map_size1);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll2, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll2, '%02d')], {});

ll2 = ll2+1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll2-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'b']}, [h_default w_default feature_map_size1 feature_map_size1], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll2, '%02d')]}, feature_map_size1);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll2, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll2, '%02d')], {});

ll2 = ll2+1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll2-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'b']}, [h_default w_default feature_map_size1 feature_map_size1], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll2, '%02d')]}, feature_map_size1);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll2, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll2, '%02d')], {});


l1 = l1 + 1; 

% net = add_LIEVEN_Haar(net, 'Haart_LIEVE03', {'LIEVE02_r01','LIEVE03_sx'}, 'Haart_LIEVE03f', [2 2 1 feature_map_size1*scale^2], scale);

net.addLayer(['Haart_LIEVE'  num2str(l1, '%02d')], dagnn.DWT2HD(), ['LIEVE' num2str(l1-1, '%02d') '_r' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_haartx'], {});

ll3 = 0;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d')], {['LIEVE' num2str(l1, '%02d') '_haartx'], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll3, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d') 'b']}, [h_default w_default feature_map_size1*4 feature_map_size2], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll3, '%02d')]}, feature_map_size2);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll3, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll3, '%02d')], {});

ll3 = ll3+1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll3-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll3, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d') 'b']}, [h_default w_default feature_map_size2 feature_map_size2], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll3, '%02d')]}, feature_map_size2);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll3, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll3, '%02d')], {});

ll3 = ll3+1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll3-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll3, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d') 'b']}, [h_default w_default feature_map_size2 feature_map_size2], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll3, '%02d')]}, feature_map_size2);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll3, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll3, '%02d')], {});

ll3 = ll3+1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll3-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll3, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d') 'b']}, [h_default w_default feature_map_size2 feature_map_size2], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll3, '%02d')]}, feature_map_size2);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll3, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll3, '%02d')], {});

ll3 = ll3+1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll3-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll3, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d') 'b']}, [h_default w_default feature_map_size2 feature_map_size2], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll3, '%02d')]}, feature_map_size2);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll3, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll3, '%02d')], {});


% net.addLayer('SUM_S02', dagnn.Sum(), {'LIEVE03_sx', 'LIEVE03_r05'}, 'SUM_S02', {});

ll3 = ll3+1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll3-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll3, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d') 'b']}, [h_default w_default feature_map_size2 feature_map_size2], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll3, '%02d')]}, feature_map_size2);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll3, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll3, '%02d')], {});

ll3 = ll3+1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll3-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll3, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll3, '%02d') 'b']}, [h_default w_default feature_map_size2 feature_map_size1*4], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll3, '%02d')]}, feature_map_size1*4);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll3, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll3, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll3, '%02d')], {});


%%% Sum Layer 
net.addLayer(['SUM_S' num2str(l1, '%02d')], dagnn.Sum(), {['LIEVE' num2str(l1, '%02d') '_haartx'], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll3, '%02d')]}, ['SUM_S' num2str(l1, '%02d')], {});

net.addLayer(['IHaart_LIEVE'  num2str(l1, '%02d')], dagnn.IWT2HD(), ['SUM_S' num2str(l1, '%02d')], ['Haart_LIEVE'  num2str(l1, '%02d')], {});

% net = add_ILIEVEN_Haar(net, 'IHaart_LIEVEP03', {'LIEVE03_r03', 'ILIEVEP03_x00'},'IHaart_LIEVEP03_f', [2 2 feature_map_size1 feature_map_size1*scale^2 ], scale);


l1 = l1 - 1;


ll2 = ll2+1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d')], {['Haart_LIEVE'  num2str(l1+1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'b']}, [h_default w_default feature_map_size1 feature_map_size1], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll2, '%02d')]}, feature_map_size1);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll2, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll2, '%02d')], {});

ll2 = ll2+1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll2-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'b']}, [h_default w_default feature_map_size1 feature_map_size1], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll2, '%02d')]}, feature_map_size1);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll2, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll2, '%02d')], {});

ll2 = ll2+1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll2-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'b']}, [h_default w_default feature_map_size1 feature_map_size1], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll2, '%02d')]}, feature_map_size1);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll2, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll2, '%02d')], {});

ll2 = ll2+1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll2-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll2, '%02d') 'b']}, [h_default w_default feature_map_size1 feature_map_size*4], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll2, '%02d')]}, feature_map_size*4);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll2, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll2, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll2, '%02d')], {});

net.addLayer(['SUM_S' num2str(l1, '%02d')], dagnn.Sum(), {['LIEVE' num2str(l1, '%02d') '_haartx'], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll2, '%02d')]}, ['SUM_S' num2str(l1, '%02d')], {});




%% 
net.addLayer(['IHaart_LIEVE'  num2str(l1, '%02d')], dagnn.IWT2HD(), ['SUM_S' num2str(l1, '%02d')], ['Haart_LIEVE'  num2str(l1, '%02d')], {});
% net = add_ILIEVEN_Haar(net, 'IHaart_LIEVEP02', {'LIEVE02_r03', 'ILIEVEP02x00'}, 'IHaart_LIEVEP02_f', [2 2 feature_map_size feature_map_size*scale^2], scale);

l1 = l1 - 1;


ll1 = ll1 + 1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d')], {['Haart_LIEVE'  num2str(l1+1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll1, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'b']}, [h_default w_default feature_map_size feature_map_size], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll1, '%02d')]}, feature_map_size);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll1, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll1, '%02d')], {});


ll1 = ll1 + 1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll1-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll1, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'b']}, [h_default w_default feature_map_size feature_map_size], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll1, '%02d')]}, feature_map_size);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll1, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll1, '%02d')], {});

ll1 = ll1 + 1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll1-1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_x' num2str(ll1, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'b']}, [h_default w_default feature_map_size feature_map_size], 1);

net = add_bnorm(net, ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')], {['LIEVE' num2str(l1, '%02d') '_x' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')]},...
    {['LIEVE' num2str(l1, '%02d') '_mean' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_var' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_mom' num2str(ll1, '%02d')]}, feature_map_size);   

net.addLayer(['LIEVE' num2str(l1, '%02d') '_relu' num2str(ll1, '%02d')], dagnn.ReLU(), ['LIEVE' num2str(l1, '%02d') '_bn' num2str(ll1, '%02d')], ['LIEVE' num2str(l1, '%02d') '_r' num2str(ll1, '%02d')], {});


ll1 = ll1 + 1;

net = add_conv(net, ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d')], {['LIEVE' num2str(l1, '%02d') '_r' num2str(ll1-1, '%02d')], 'prediction_res_multi'},...
    {['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'f'],  ['LIEVE' num2str(l1, '%02d') '_layer' num2str(ll1, '%02d') 'b']}, [h_default w_default feature_map_size 4], 1);
        
net.addLayer('Prediction', dagnn.Sum(), {['LIEVE' num2str(l1, '%02d') '_t'], 'prediction_res_multi'}, 'prediction_multi');  
    
% net = add_ILIEVEN_Haar(net, 'IHaart_LIEVEP01', {'prediction_multi', 'prediction'}, 'IHaart_LIEVEP01_f', [2 2 1 scale^2], scale);        
net.addLayer('IHaart_LIEVEP01', dagnn.IWT2HD(), 'prediction_multi', 'prediction', {});

%   %% -------------------------------- SUM and Loss --------------------------------------

      
net.addLayer('objective', dagnn.Loss('loss', 'l2'), {'prediction','label'}, 'objective') ;       
               
net.vars(net.getVarIndex('prediction')).precioILIEVE = 1 ;

end


% --------------------------------------------------------------------
function net = add_bnorm(net, layername, varname, parname, out)
% --------------------------------------------------------------------


meanvar  =  [zeros(out,1,'single'),0.01*ones(out,1,'single')];

net.addLayer(layername, ...
    dagnn.BatchNorm(), ...
    varname{1}, varname{2}, {parname{1},parname{2},parname{3}});

f = net.getParamIndex(parname{1}) ;net.params(f).learningRate = 1;
net.params(f).value = clipp(sqrt(2/(9*out))*randn(out,1,'single'),0.03) ;net.params(f).weightDecay  = 0 ;
f = net.getParamIndex(parname{2}) ;net.params(f).learningRate = 1;
net.params(f).value = zeros(out,1,'single') ;net.params(f).weightDecay  = 0 ;
f = net.getParamIndex(parname{3}) ;net.params(f).learningRate = 1;
net.params(f).value = meanvar ;net.params(f).weightDecay  = 0 ;


end

% --------------------------------------------------------------------
function net = add_conv(net, layername, varname, parname, filter_size,dilatefactor)
% --------------------------------------------------------------------

    stride_default = 1;

    pad_default = dilatefactor;
    % filtername = sprintf('%s%s', layername, 'f');
    % biasname   = sprintf('%s%s', name, 'b');
    % if ~isfield(net, 'momentum')
    %   net.momentum = net.params ;
    % end
    if numel(parname) == 2
        net.addLayer(layername, ...
            dagnn.Conv('pad', pad_default,'stride', stride_default,'dilate',dilatefactor), ...
            varname{1}, varname{2}, {parname{1},parname{2}});

        f = net.getParamIndex(parname{1}) ;
        net.params(f).value = init_weight(filter_size, 'single') ;

        net.params(f).learningRate = 1;
        net.params(f).weightDecay  = 1;
        net.params(f).trainMethod = 'adam';
        %     net.params(f).t = 1;
        f = net.getParamIndex(parname{2}) ;
        net.params(f).value = zeros(filter_size(4), 1, 'single');
        if filter_size(4) == 1 && filter_size(3) == 3
            value = ones(filter_size(4))*16/255;
            net.params(f).value = single(value);
        end
        net.params(f).learningRate = 1;
        net.params(f).weightDecay  = 0;
        net.params(f).trainMethod = 'adam';
        %     net.params(f).t = 1;
    else

        net.addLayer(layername, ...
            dagnn.Conv('pad', pad_default,'stride', stride_default,'hasBias',false,'dilate',dilatefactor), ...
            varname{1}, varname{2}, parname{1});

        f = net.getParamIndex(parname{1}) ;
        net.params(f).value = init_weight(filter_size, 'single') ;
        net.params(f).learningRate = 1;
        net.params(f).weightDecay  = 1;
        net.params(f).trainMethod = 'adam';
        %     net.params(f).t = 1;
    end

end


% -------------------------------------------------------------------------
function weights = init_weight(filter_size,type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.
h   = filter_size(1);
w   = filter_size(2);
in  = filter_size(3);
out = filter_size(4);


sc = sqrt(1/(h*w*out));

if filter_size(4) == 1
    sc = sqrt(1/(h*w*in));
end

weights = randn(h, w, in, out, type)*sc ;

weights = orthrize(weights);

end


function A = orthrize(A)
B = A;

A = reshape(A,[size(A,1)*size(A,2)*size(A,3),size(A,4),1,1]);
if size(A,1)> size(A,2)
    [U,S,V] = svd(A,0);
else
    [U,S,V] = svd(A,'econ');
end

S1 =ones(size(diag(S)));
A = U*diag(S1)*V';
A = reshape(A,size(B));

end

function A = clipp(A,b)

A(A>=0&A<b)=b;
A(A<0&A>-b)=-b;
end
