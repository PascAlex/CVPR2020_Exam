%% Clear everything
clear all
close all
clc

%% Transfer learning
%Download it form adds-on link
% By default should download imagenet weights
net = alexnet;

%% Data Preprocessing
%The input layer take [227 227 3]-sized images
project = fullfile('dataset','train');

imds = imageDatastore(project, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

imds.ReadFcn = @(x)repmat(imresize(imread(x),[227 227]), 1,1,3);

%% Only cropping

cropped_project = fullfile('dataset','train_cropped');

imdsCropped2 = imageDatastore(cropped_project, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

imdsCropped2.ReadFcn = @(x)repmat(imresize(imread(x),[227 227]), 1,1,3);
%% Only Reflection 

reflected_project = fullfile('dataset','train_reflected');

imdsReflected2 = imageDatastore(reflected_project, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

imdsReflected2.ReadFcn = @(x)repmat(imresize(imread(x),[227 227]), 1,1,3);
%% Splitting
quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');

%% Data augmentation

imdsAugmented = imageDatastore(cat(1,imdsTrain.Files,imdsReflected2.Files));
imdsAugmented.Labels = cat(1,imdsTrain.Labels, imdsReflected2.Labels);

imdsAugmented.ReadFcn = @(x)repmat(imresize(imread(x),[227 227]), 1,1,3);
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
           
           fullyConnectedLayer(15, 'WeightLearnRateFactor', 2, ...
                                'BiasLearnRateFactor', 5)
           
           softmaxLayer
           
           classificationLayer];
       
%% Options
options = trainingOptions('adam', ...    
    'ValidationData',imdsValidation, ...
    'MaxEpochs', 30, ...
    'InitialLearnRate', 0.001, ...
    'ValidationPatience',5 ,...
    'Verbose',false, ...
    'Shuffle', 'every-epoch', ...
    'MiniBatchSize',64, ... 
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress');

%% Train the network

net_trained = trainNetwork(imdsAugmented, layers, options);

%%
%Look at the output + accuracy

YPredicted = classify(net_trained,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest)

%% confusion matrix
figure
plotconfusion(YTest,YPredicted)

%Accuracy finale: 86% (84/85 random cropping) 

%% SVM

%Feature Extraction

feature_train = activations(net, imdsTrain, 'fc6', 'OutputAs', 'rows'); 
feature_test = activations(net, imdsTrain, 'fc6', 'OutputAs', 'rows');

%Extract labels
YTrain1 = imdsTrain.Labels;
YTest1 = imdsTest.Labels;

%Classifier
classifier = fitcecoc(feature_train, YTrain1);
YPred1 = predict(classifier, feature_test);

%Accuracy
accuracy1 = mean(YPred1 == YTest1)

% acc = 0.73 conv5
% acc = 0.87 fc6 0.84 w data augmenter
% acc = 0.86 fc7 0.86 w data augmenter
% acc = 0.84 fc8 0.84 w data augmenter

%% plot confusion
figure
plotconfusion(YTest1,YPred1)