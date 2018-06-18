close all
clear all
clc

addpath('../MRF');
addpath('../');
%MRFParams = single([105 200 1.0]);% Shanghaitech Part_A
MRFParams = single([200 200 8]);% Shanghaitech Part_B

load('processed/predictions.mat');
load('processed/raw_input.mat');

read_path = 'images/';
store_path = 'output/';

n = numel(features);
k = 1;

for i = 1 : n
    disp(i);
    patch = features{i};
    
    height = size(patch,1);
    width = size(patch,2);

    p = reshape(predictions(k: k + height * width - 1), width, height);
    k = k + height * width;
    
    % The marginal data of the predicted count matrix is 0 after applying MRF, 
    % so first extending the predicted count matrix by copying marginal data.
    p = uint8(p)';
    p = [p(1,:); p];
    p = [p ;p(end,:)];
    p = [p(:, 1) p];
    p = [p p(:, end)];
    % apply MRF
    p = MRF(p, MRFParams);
    p = p(2:end-1, 2: end-1);
    
    finalcount = FinalCount(p);
    
    im = imread([read_path 'IMG_' num2str(i) '.jpg']);
    figure
    imshow(im)
    title(['Predicted count : ' num2str(finalcount)])
    %savefig([store_path 'IMG_' num2str(i) '.fig']);
    saveas(gcf, [store_path 'IMG_' num2str(i) '.png'])
    close
end
