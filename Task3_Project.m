%% Clear everything
clear all
close all
clc

%% Transfer learning
%Download it form adds-on link
% By defaukt should download imagenet weights
net = alexnet;

%% Data Preprocessing
%The input layer take [227 227 3]-sized images

project = fullfile('dataset','train');

imds = imageDatastore(project, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

imds.ReadFcn = @(x)repmat(imresize(imread(x),[227 227]), 1,1,3);

%% Splitting
quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');

%% Data Augmentation
augmenter = imageDataAugmenter(...
            'RandXReflection', true);
augmented_train = augmentedImageDatastore([227 227 3], imdsTrain, 'DataAugmentation', augmenter);
             
%% Resize test set
project_test  = fullfile('dataset','test');

imdsTest = imageDatastore(project_test, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)repmat(imresize(imread(x),[227 227]),1,1,3);

%% Start transfer learning

% First freeze the weights in all layers except the last fullyconnected one
% freeze weights = force WeightLearnRateFactor to be 0

%tha last fully connected is in position 23 of 25
layers_to_transfer = net.Layers(1:end-3);
for i=1:numel(layers_to_transfer)
   if(isprop(net.Layers(i), 'WeightLearnRateFactor')) %check if that property exists
       layers_to_transfer(i).WeightLearnRateFactor = 0;
   end    
end    

%% Layers
%train the network and fine tune the last weights

layers = [layers_to_transfer
           %parametri messi a caso , DA CAMBIARE
           fullyConnectedLayer(15, 'WeightLearnRateFactor', 2, ...
                                'BiasLearnRateFactor', 5)
           
           softmaxLayer
           
           classificationLayer];
       
%% Options
options = trainingOptions('adam', ...    
    'ValidationData',imdsValidation, ...
    'MaxEpochs', 50, ...
    'InitialLearnRate', 0.0001, ...
    'ValidationPatience',5,...
    'Verbose',false, ...
    'Shuffle', 'every-epoch', ...
    'MiniBatchSize',64, ... %provare con 128
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress');

%% Train the network

net_trained = trainNetwork(augmented_train, layers, options);

%%
%Look at the output + accuracy

YPredicted = classify(net_trained,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest)

%% confusion matrix
figure
plotconfusion(YTest,YPredicted)

% accuracy around 85%

%% SVM

%Feature Extraction

feature_train = activations(net, augmented_train, 'fc7', 'OutputAs', 'rows');
feature_test = activations(net, imdsTest, 'fc7', 'OutputAs', 'rows');

%Extract labels
YTrain1 = imdsTrain.Labels;
YTest1 = imdsTest.Labels;

%Classifier
classifier = fitcecoc(feature_train, YTrain1);
YPred1 = predict(classifier, feature_test);

%Accuracy
accuracy1 = mean(YPred1 == YTest1)

%accuracy around 83/84%
