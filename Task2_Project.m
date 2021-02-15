close all force

project = fullfile('dataset','train');

imds = imageDatastore(project, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% split in training and validation sets: 85% - 15%
quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');


%% 2.1 data augmentation

% it has been decided to do it by hand in order to perform the random
% cropping as described in the project assignments; it was not possible to
% obtain a rondom cropping window by following the imageDataAugmenter
% approach (there would have been a window of fixed size).
% The function used for the cropping procedure (randCrop) is defined at the end
% of this script


% create the transformed datasets and write them in the disk

imdsCropped = transform(imdsTrain,@(x) randCrop(x));
imdsReflected = transform(imdsTrain,@(x) flip(x, 2));

location1 = 'C:\CroppedDataStore';
writeall(imdsCropped, location1, 'OutputFormat', 'jpg', 'FilenamePrefix', 'cropped_')

location2 = 'C:\ReflectedDataStore';
writeall(imdsCropped, location2, 'OutputFormat', 'jpg', 'FilenamePrefix', 'reflected_')

% create the augmented image datastore

imdsCropped2 = imageDatastore( 'C:\CroppedDataStore', ... 
    'IncludeSubfolders',true,'LabelSource','foldernames');

imdsReflected2 = imageDatastore( 'C:\ReflectedDataStore', ... 
    'IncludeSubfolders',true,'LabelSource','foldernames');


imdsAugmented = imageDatastore(cat(1,imdsTrain.Files,imdsCropped2.Files, imdsReflected2.Files));
imdsAugmented.Labels = cat(1,imdsTrain.Labels,imdsCropped2.Labels, imdsReflected2.Labels);

% automatic resizing
imdsValidation.ReadFcn = @(x)imresize(imread(x),[64 64]);
imdsAugmented.ReadFcn = @(x)imresize(imread(x),[64 64]);


% same layers and option from the task1

layers = [
    imageInputLayer([64 64 1],'Name','input') 
    
    convolution2dLayer(3,8,'Name','conv_1','WeightsInitializer','narrow-normal') 
    
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(3,16,'Name','conv_2','WeightsInitializer','narrow-normal')
    
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(3,32,'Name','conv_3','WeightsInitializer','narrow-normal') 
   
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(15,'WeightsInitializer','narrow-normal','Name','fc_1')
    
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')];

    lgraph = layerGraph(layers);
    analyzeNetwork(lgraph)
    
options = trainingOptions('sgdm', ...
    'ValidationData',imdsValidation, ...
    'InitialLearnRate', 0.001, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 2,...
    'Verbose',false, ...
    'MiniBatchSize',32, ...
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress');

% train the network with the new data 

net = trainNetwork(imdsAugmented,layers,options);

% evaluate performance on test set

project_test  = fullfile('dataset','test');

imdsTest = imageDatastore(project_test, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

% apply the network to the test set

YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy1 = sum(YPredicted == YTest)/numel(YTest);

% confusion matrix
figure
plotconfusion(YTest,YPredicted);

% accuracy = 0.3819

%% 2.2 add batch normalization layers

layers = [
    imageInputLayer([64 64 1],'Name','input') 
    
    convolution2dLayer(3,8,'Name','conv_1','WeightsInitializer','narrow-normal')
    
    batchNormalizationLayer('Name', 'BN_1')
    
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(3,16,'Name','conv_2', 'WeightsInitializer','narrow-normal')
    
    batchNormalizationLayer('Name', 'BN_2')
    
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(3,32,'Name','conv_3', 'WeightsInitializer','narrow-normal')
    
    batchNormalizationLayer('Name', 'BN_3')
   
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(15,'WeightsInitializer','narrow-normal','Name','fc_1')
    
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')];

    lgraph = layerGraph(layers);
    analyzeNetwork(lgraph)
   
% training
    
net = trainNetwork(imdsAugmented,layers,options);

% evaluate performance on test set

project_test  = fullfile('dataset','test');

imdsTest = imageDatastore(project_test, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

% apply the network to the test set

YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy2 = sum(YPredicted == YTest)/numel(YTest);

% confusion matrix
figure
plotconfusion(YTest,YPredicted);

% accuracy = 0.4704

%% 2.3 change filters' size (3*3, 5*5, 7*7)

layers = [
    imageInputLayer([64 64 1],'Name','input') 
    
    convolution2dLayer(3,8,'Name','conv_1','WeightsInitializer','narrow-normal')
    
    batchNormalizationLayer('Name', 'BN_1')
    
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(5,16,'Name','conv_2','WeightsInitializer','narrow-normal')
    
    batchNormalizationLayer('Name', 'BN_2')
    
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(7,32,'Name','conv_3','WeightsInitializer','narrow-normal')
    
    batchNormalizationLayer('Name', 'BN_3')
   
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(15,'WeightsInitializer','narrow-normal','Name','fc_1')
    
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')];

    lgraph = layerGraph(layers);
    analyzeNetwork(lgraph)

% training

net = trainNetwork(imdsAugmented,layers,options);

% evaluate performance on test set

project_test  = fullfile('dataset','test');

imdsTest = imageDatastore(project_test, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

% apply the network to the test set

YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy3 = sum(YPredicted == YTest)/numel(YTest);

% confusion matrix
figure
plotconfusion(YTest,YPredicted);

% accuracy = 0.4766

%% 2.4 play with the optimization parameters

% switch to the adam optimizer
% minibach size = 64

options = trainingOptions('adam', ...
'InitialLearnRate', 0.001, ...
'ValidationData',imdsValidation, ...
'ValidationFrequency', 50, ...
'ValidationPatience', 2,...
'Verbose',false, ...
'MiniBatchSize',64, ...
'ExecutionEnvironment','parallel',...
'Plots','training-progress');

% training

net = trainNetwork(imdsAugmented,layers,options);

% evaluate performance on test set

project_test  = fullfile('dataset','test');

imdsTest = imageDatastore(project_test, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

% apply the network to the test set

YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy4 = sum(YPredicted == YTest)/numel(YTest);

% confusion matrix
figure
plotconfusion(YTest,YPredicted);

% accuracy = 0.5002

%% 2.5 dropout layers

layers = [
    imageInputLayer([64 64 1],'Name','input') 
    
    convolution2dLayer(3,8,'Name','conv_1','WeightsInitializer','narrow-normal')
    
    batchNormalizationLayer('Name', 'BN_1')
    
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    dropoutLayer(.1 , 'Name','dropout_1')
    
    convolution2dLayer(5,16,'Name','conv_2','WeightsInitializer','narrow-normal')
    
    batchNormalizationLayer('Name', 'BN_2')
    
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    dropoutLayer(.1 , 'Name','dropout_2')
    
    convolution2dLayer(7,32,'Name','conv_3','WeightsInitializer','narrow-normal')
    
    batchNormalizationLayer('Name', 'BN_3')
   
    reluLayer('Name','relu_3')
    
    dropoutLayer(.1 , 'Name','dropout_3')
    
    fullyConnectedLayer(15,'WeightsInitializer','narrow-normal','Name','fc_1')
    
    dropoutLayer(.5 , 'Name','dropout_4')
    
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')];

    lgraph = layerGraph(layers); % to run the layers need a name
    analyzeNetwork(lgraph)

% training

net = trainNetwork(imdsAugmented,layers,options);

% evaluate performance on test set

project_test  = fullfile('dataset','test');

imdsTest = imageDatastore(project_test, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

% apply the network to the test set

YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy5 = sum(YPredicted == YTest)/numel(YTest);

% confusion matrix
figure
plotconfusion(YTest,YPredicted);

% accuracy = 0.5203
% clearly less overfitting

%% 2.6 ensemble of CNNs

% define how many networks you want to train

nnets = 10;

% train nnets models from the previous point
% the predict funtion returns a matrix with the scores computed for each
% observation. Then we compute the sum of the scores and finally we devide by
% the number of models in order to obtain the averages of the scores

scores = zeros(numel(YTest),15);

for i = 1:nnets
   net = trainNetwork(imdsAugmented,layers,options);
   scores = scores + predict(net, imdsTest);   
end

avscores = scores/nnets;

% obtain the indexes of the predicted class

[val, predInd] = max(avscores,[],2);

rightpred = 0;

% ensemble accuracy

for i= 1:numel(YTest)
    
    for j = 1:15
       
       if net.Layers(end).Classes(j) == YTest(i)
           classInd = j;
       end
       
    end
    
    if predInd(i) == classInd
        rightpred = rightpred + 1;
    end
    
end

ensembleAccuracy = rightpred/numel(YTest);

% accuracy = 0.5843

% create a cropping function so that it's random also in the choice of the
% output window size

function [resizedCroppedImage] = randCrop(originalImage)
    a = size(originalImage);
    MinSizeH = fix((a(2)/3));
    MaxSizeH = (a(2));
    MinSizeL = fix(a(1)/3);
    MaxSizeL = a(1);
    rH = fix((MaxSizeH - MinSizeH).*rand(1,1) + MinSizeH);
    rL = fix((MaxSizeL - MinSizeL).*rand(1,1) + MinSizeL);
    targetSize = [rL, rH];
    win = randomCropWindow2d(a,targetSize);
    croppedImage = imcrop(originalImage,win);
    resizedCroppedImage = imresize(croppedImage, [64,64]);
 end
