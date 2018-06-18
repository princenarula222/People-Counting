function [] = Extract(imagesPath, label)

%% set images path and ground-truth path
filePattern = fullfile(imagesPath, '/*.jpg');
ImageFiles = dir(filePattern);
n = length(ImageFiles) %number of images
features = cell(1, n);

winSize = 100;
winStep = winSize - 1;
xStep = 50;
yStep = 50;


%% init pre-trained resnet-152 model 
net = dagnn.DagNN.loadobj(load('..\imagenet-resnet-152-dag.mat'));
net.mode = 'test';
net.conserveMemory = 0;

%% 
for i = 1 : n
    disp(i)
    im = imread([imagesPath 'IMG_' num2str(i) '.jpg']);
    [height, width, channel] = size(im);
    
    newHeight = 300;
    newWidth = 300;
    %newHeight = round(height/50)*50;
    %newWidth = round(width/50)*50;

    im = imresize(im, [newHeight, newWidth]);
    if channel == 1
        tmp = zeros(newHeight, newWidth, 3);
        tmp(:, :, 1) = im;
        tmp(:, :, 2) = im;
        tmp(:, :, 3) = im;
        im = tmp;
    end
    
    
    y = 1;
    row = 1;
    patchFeature = zeros(newHeight / 50 - 1, newWidth/50 - 1, 1000);
    while(y + winStep  <= newHeight)
        x = 1;
        column = 1;
        while(x + winStep <= newWidth)
            img = im(y:y + winStep, x: x + winStep, :); % get image patch
            img = single(img);
            
            im_ = imresize(img, net.meta.normalization.imageSize(1:2));
            im_ = im_ - net.meta.normalization.averageImage ;
            
            net.eval({'data', im_});
            patchFeature(row ,column ,:) = reshape(net.vars(net.getVarIndex('fc1000')).value, 1, 1000);
            
            x = x + xStep;
            column = column + 1;
        end
        y = y + yStep;
        row = row + 1;
    end
    
    features{i} = patchFeature;
end
save (['processed\' label '.mat'], 'features')

