## Copyright (C) 2007 Michel D. Schmid  <michaelschmid@users.sourceforge.net>
##
##
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2, or (at your option)
## any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.

## author: msd 



## load data

mData = load("mData.txt","mData");
# mData = mData.mData(1:10,:);
mData = mData.mData;
[nRows, nColumns] = size(mData);
# this file contains 13 columns. The first 12 columns are the inputs
# the last column is the output
# remove column 4, 8 and 12
# 89 rows

## first permute the whole matrix in row wise
## this won't be used right now for debug and test purpose
# order = randperm(nRows);
# mData(order,:) = mData;

mOutput = mData(:,end);
mInput = mData(:,1:end-1);
mInput(:,[4 8 12]) = []; # delete column 4, 8 and 12

## now prepare data
mInput = mInput';
mOutput = mOutput';
mOutput = [mOutput; mOutput*4; mOutput*10];
# now split the data matrix in 3 pieces, train data, test data and validate data
# the proportion should be about 1/2 train, 1/3 test and 1/6 validate data
# in this neural network we have 12 weights, for each weight at least 3 train sets..
# (that's a rule of thumb like 1/2, 1/3 and 1/6)
# 1/2 of 89 = 44.5; let's take 44 for training
nTrainSets = floor(nRows/2);
# now the rest of the sets are again 100%
# ==> 2/3 for test sets and 1/3 for validate sets
nTestSets = (nRows-nTrainSets)/3*2;
nValiSets = nRows-nTrainSets-nTestSets;

mValiInput = mInput(:,1:nValiSets);
mValliOutput = mOutput(:,1:nValiSets);
mInput(:,1:nValiSets) = [];
mOutput(:,1:nValiSets) = [];
mTestInput = mInput(:,1:nTestSets);
mTestOutput = mOutput(:,1:nTestSets);
mInput(:,1:nTestSets) = [];
mOutput(:,1:nTestSets) = [];
mTrainInput = mInput(:,1:nTrainSets);
mTrainOutput = mOutput(:,1:nTrainSets);

[mTrainInputN,cMeanInput,cStdInput] = prestd(mTrainInput);# standardize inputs

## comments: there is no reason to standardize the outputs because we have only
# one output ...

# define the max and min inputs for each row
mMinMaxElements = min_max(mTrainInputN); % input matrix with (R x 2)...

## define network
nHiddenNeurons = 2;
nOutputNeurons = 3;

MLPnet = newff(mMinMaxElements,[nHiddenNeurons nOutputNeurons],{"logsig","purelin"},"trainlm","learngdm","mse");
## for test purpose, define weights by hand
MLPnet.IW{1,1}(1,:) = 1.5;
MLPnet.IW{1,1}(2,:) = 0.5;
MLPnet.LW{2,1}(1,:) = 1.5;
MLPnet.LW{2,1}(2,:) = 0.5;
MLPnet.LW{2,1}(3,:) = 0.1;
MLPnet.b{1,1}(1,:) = 0.5;
MLPnet.b{1,1}(2,:) = 1.5;
MLPnet.b{2,1}(1,:) = 0.5;
MLPnet.b{2,1}(2,:) = 0.5;
MLPnet.b{2,1}(3,:) = 0.1;

# saveMLPStruct(MLPnet,"MLP3test.txt");
#disp("network structure saved, press any key to continue...")
#pause

## define validation data new, for MATLAB(TM) compatibility
VV.P = mValiInput;
VV.T = mValliOutput;

## standardize also the validate data
VV.P = trastd(VV.P,cMeanInput,cStdInput);

#[net,tr,out,E] = train(MLPnet,mInputN,mOutput,[],[],VV);
MLPnet.trainParam.show = NaN;
[net] = train(MLPnet,mTrainInputN,mTrainOutput,[],[],VV);
# saveMLPStruct(net,"MLP3testNachTraining.txt");
# disp("network structure saved, press any key to continue...")
# disp("nach Training")
# pause

# % make preparations for net test and test MLPnet
#     % standardise input & output test data
[mTestInputN] = trastd(mTestInput,cMeanInput,cStdInput);

#     % simulate net
[simOut] = sim(net,mTestInputN);%,Pi,Ai,mTestOutput);