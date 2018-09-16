function [data, labels, set] = patches_generation(modelname,size_input,size_label,stride,folder,mode)
padding = abs(size_input - size_label)/2;
ss = 1;
ext               =  {'*.jpg','*.png','*.bmp','*.jpeg', '*.JPEG'};

stride1 = 100;
stride2 = 100;
count_num = 0;
data = [];
labels = [];
set = [];
for fi = 1:numel(folder)
    filepaths           =  [];
    for i = 1 : length(ext)
        filepaths = [filepaths; dir(fullfile(folder{fi}, ext{i}))];
    end
    if fi == 8
        start_frm = 1;
    else
        start_frm = 1;
    end
    for i = start_frm : length(filepaths)
        count = 0;
        fprintf('Folder %d Image %d\n',fi, i);
        try
            image= imread(fullfile(folder{fi},filepaths(i).name));
            if size(image,3) ==3
                %% For denoising
                image = rgb2gray(image);
                %% For SISR and deblocking
%                  image = rgb2ycbcr(image);
            end
        catch
            continue;
        end 
        im_label = im2single(image(:, :, 1));
        [hei, wid] = size(im_label);
        if hei < size_label || wid < size_label
            continue;
        end
        
        for x = 1 : stride1 : (hei-size_input+1)
            for y = 1 :stride2 : (wid-size_input+1)
                subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);
                count = count+1;  
                imwrite(subim_label, ['./' modelname '/' num2str(fi, '%d') num2str(i, '%04d') num2str(count, '%03d') '.png']);
            end
        end
%        count_num = count_num + count;
        fprintf('Folder %d count %d\n', fi, count);
    end
end
%quit;


