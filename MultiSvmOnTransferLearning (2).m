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
%% Resize test set
project_test  = fullfile('dataset','test');

imdsTest = imageDatastore(project_test, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)repmat(imresize(imread(x),[227 227]),1,1,3);
%% SVM

%Feature Extraction 6 layer
feature_train = activations(net, imds, 'fc6', 'OutputAs', 'rows'); 
feature_test = activations(net, imdsTest, 'fc6', 'OutputAs', 'rows');

%Feature Extraction 7 layer
feature_train7 = activations(net, imds, 'fc7', 'OutputAs', 'rows'); 
feature_test7 = activations(net, imdsTest, 'fc7', 'OutputAs', 'rows');


%Extract labels
YTrain1 = imds.Labels;
YTest1 = imdsTest.Labels;
%% MultiSvm 6 Layer
numeric_train_label = grp2idx(YTrain1);
numeric_test_label = grp2idx(YTest1);

results = multisvm(feature_train, numeric_train_label, feature_test);
accuracy1 = mean(results == numeric_test_label)

%%
%Convert back to categoric variables
valueset = [1:15];
catnames = {'Bedroom','Coast','Forest','Highway','Industrial','InsideCity',...
    'Kitchen','LivingRoom','Mountain','Office','OpenCountry','Store',...
    'Street','Suburb','TallBuilding'};
predictedValues = categorical(results,valueset,catnames);
%Plot Confusion Matrix
figure
plotconfusion(YTest1,predictedValues)

 %% MultiSvm 7 Layer
% numeric_train_label = grp2idx(YTrain1);
% numeric_test_label = grp2idx(YTest1);
% 
% results = multisvm(feature_train7, numeric_train_label, feature_test7);
% accuracy2 = mean(results == numeric_test_label)
% 
% %Convert back to categoric variables
% valueset = [1:15];
% catnames = {'Bedroom','Coast','Forest','Highway','Industrial','InsideCity',...
%     'Kitchen','LivingRoom','Mountain','Office','OpenCountry','Store',...
%     'Street','Suburb','TallBuilding'};
% predictedValues = categorical(results,valueset,catnames);
% %Plot Confusion Matrix
% figure
% plotconfusion(YTest1,predictedValues)

