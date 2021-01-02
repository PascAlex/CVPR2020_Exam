close all force

project = fullfile('dataset','train');

imds = imageDatastore(project, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% automatic resizing
imds.ReadFcn = @(x)imresize(imread(x),[64 64]);

% split in training and validation sets: 85% - 15%
quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');

%% data augmentation

% left/right reflection

augmenter = imageDataAugmenter(...
    'RandXReflection', true)

auimdsTrain = augmentedImageDatastore([64 64 1],imdsTrain,'DataAugmentation',augmenter);

% same layers and option from the task1

layers = [
    imageInputLayer([64 64 1],'Name','input','Normalization','zscore') 
    
    convolution2dLayer(3,8,'Name','conv_1') 
    
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(3,16,'Name','conv_2')
    
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(3,32,'Name','conv_3') 
   
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(15,'WeightsInitializer','narrow-normal','Name','fc_1')
    
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')];

    lgraph = layerGraph(layers);
    analyzeNetwork(lgraph)
    
options = trainingOptions('sgdm', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 3,...
    'Verbose',false, ...
    'MiniBatchSize',32, ...
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress')

% train the network with the new data 

net = trainNetwork(auimdsTrain,layers,options);

% evaluate performance on test set

project_test  = fullfile('dataset','test');

imdsTest = imageDatastore(project_test, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

% apply the network to the test set

YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest)

% confusion matrix
figure
plotconfusion(YTest,YPredicted);

% accuracy = 0.44

%% add batch normalization layers

layers = [
    imageInputLayer([64 64 1],'Name','input','Normalization','zscore') 
    
    convolution2dLayer(3,8,'Name','conv_1')
    
    batchNormalizationLayer('Name', 'BN_1')
    
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(3,16,'Name','conv_2')
    
    batchNormalizationLayer('Name', 'BN_2')
    
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(3,32,'Name','conv_3')
    
    batchNormalizationLayer('Name', 'BN_3')
   
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(15,'WeightsInitializer','narrow-normal','Name','fc_1')
    
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')];

    lgraph = layerGraph(layers);
    analyzeNetwork(lgraph)
   
% training
    
net = trainNetwork(auimdsTrain,layers,options);

% evaluate performance on test set

project_test  = fullfile('dataset','test');

imdsTest = imageDatastore(project_test, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

% apply the network to the test set

YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest)

% confusion matrix
figure
plotconfusion(YTest,YPredicted);

% accuracy = 0.48

%% change filters' size (3*3, 5*5, 7*7)

layers = [
    imageInputLayer([64 64 1],'Name','input','Normalization','zscore') 
    
    convolution2dLayer(3,8,'Name','conv_1')
    
    batchNormalizationLayer('Name', 'BN_1')
    
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(5,16,'Name','conv_2')
    
    batchNormalizationLayer('Name', 'BN_2')
    
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer(7,32,'Name','conv_3')
    
    batchNormalizationLayer('Name', 'BN_3')
   
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(15,'WeightsInitializer','narrow-normal','Name','fc_1')
    
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')];

    lgraph = layerGraph(layers);
    analyzeNetwork(lgraph)

% training

net = trainNetwork(auimdsTrain,layers,options);

% evaluate performance on test set

project_test  = fullfile('dataset','test');

imdsTest = imageDatastore(project_test, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

% apply the network to the test set

YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest)

% confusion matrix
figure
plotconfusion(YTest,YPredicted);

% accuracy = 0.50

%% play with the optimization parameters

% switch to the adam optimizer
% learning rate = 0.001
% minibach size = 64

options = trainingOptions('adam', ...
'InitialLearnRate', 0.001, ...
'ValidationData',imdsValidation, ...
'MaxEpochs', 30, ...
'ValidationFrequency', 50, ...
'ValidationPatience', 3,...
'Verbose',false, ...
'MiniBatchSize',64, ...
'ExecutionEnvironment','parallel',...
'Plots','training-progress');

% training

net = trainNetwork(auimdsTrain,layers,options);

% evaluate performance on test set

project_test  = fullfile('dataset','test');

imdsTest = imageDatastore(project_test, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

% apply the network to the test set

YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest)

% confusion matrix
figure
plotconfusion(YTest,YPredicted);

% accuracy = 0.52

%% dropout layers

layers = [
    imageInputLayer([64 64 1],'Name','input','Normalization','zscore') 
    
    convolution2dLayer(3,8,'Name','conv_1')
    
    batchNormalizationLayer('Name', 'BN_1')
    
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    dropoutLayer(.1 , 'Name','dropout_1')
    
    convolution2dLayer(5,16,'Name','conv_2')
    
    batchNormalizationLayer('Name', 'BN_2')
    
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    dropoutLayer(.1 , 'Name','dropout_2')
    
    convolution2dLayer(7,32,'Name','conv_3')
    
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

net = trainNetwork(auimdsTrain,layers,options);

% evaluate performance on test set

project_test  = fullfile('dataset','test');

imdsTest = imageDatastore(project_test, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

% apply the network to the test set

YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest)

% confusion matrix
figure
plotconfusion(YTest,YPredicted);

% accuracy = 0.54
% clearly less overfitting

%% ensemble of CNNs

% define how many networks you want to train

nnets = 10;

% train nnets models from the previous point
% the predict funtion returns a matrix with the scores computed for each
% observation. Then we compute the sum of the scores and finally we devide by
% the number of models in order to obtain the averages of the scores

scores = zeros(numel(YTest),15);

for i = 1:nnets
   net = trainNetwork(auimdsTrain,layers,options);
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

% accuracy = 0.60
