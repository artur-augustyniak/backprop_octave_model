% Dodaję ścieżkę biblioteki 
% TODO
% niestety w tym momencie
% trzeba parametryzować funkcje transferu bezpośrednio
% w plikach biblioteki

	addpath("./nnet");

% Dane http://archive.ics.uci.edu/ml/datasets/Wall-Following+Robot+Navigation+Data

	mData = load("./datasets/shuffle_robot_short.data");
	numClasses = 4;	
	
% Konwersja wyjścia dla zmiennej nominalnej/klasyfikacji.
% 4 klasy
% -- Move-Forward	[1 0 0 0]
% -- Slight-Right-Turn	[0 1 0 0]
% -- Sharp-Right-Turn	[0 0 1 0]
% -- Slight-Left-Turn	[0 0 0 1]

	[nRows, nColumns] = size(mData);
	mInput = mData(:,1:end-1);
	mOutput = mData(:,end:end);

	vOutput = zeros(nRows, numClasses);
	
	vOutput_temp = zeros(numClasses, nRows);
	tempVect = 0:numClasses:(numClasses*nRows-1);
	vOutput_temp(tempVect'+mOutput)=1;
	vOutput = vOutput_temp';

	
% + macierz wygenerowana na podstawie identyfikatorów klas

	mData = [mInput, vOutput];

% nowe wymiary

	[nRows, nColumns] = size(mData);
	
% Rand - kolejnośc encji zbioru uczącego wyłączona do porównań
	%order = randperm(nRows);
	%mData(order,:) = mData;
	
% Tniemy wejście

	mInput = mData(:,1:end-numClasses);
	mOutput = mData(:,end-(numClasses-1):end);		
			
% wycięcie podanych kolumn

	%mInput(:,[4 8 12]) = []; 


% preprocess

	mInput = mInput';
	mOutput = mOutput';

% Dzielę macierz na 3 części
%     dane treningowe
%     dane testowe (egzamin)
%     dane walidujące 
% Proporcje 
%    1/2 trening
%    1/3 test
%    1/3 walidacja
% Ta siec ma 436 wag, minimalny zbiór uczący 4360 el /1.222 ~ 4464

	nTrainSets = floor(nRows/1.222);

% reszta 100%
% ==> 2/3 dla testów i 1/3 dla walidacji

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

% Normalizacja wejść

	[mTrainInputN,cMeanInput,cStdInput] = prestd(mTrainInput);

% min max dla każdego wiersza (MATLAB)(R x 2)

	mMinMaxElements = min_max(mTrainInputN);

% opis sieci ilość neuronów dobrana wg zależności wynikającej ze średniej geometrycznej długości wektora we i wy (lub teorii Kołmogorowa Ossowski s98)
% wynagane 16 neuronów warstw ukrytych oszacowanie VCDim dla takiej sieci (2* ilośc wag ponieważ funkcja tranferu to sigmoida) ~ 436

	nHiddenNeurons = 9;
	n2HiddenNeurons = 7;
	nOutputNeurons = numClasses;

	MLPnet = newff(
		mMinMaxElements,
		[nHiddenNeurons	n2HiddenNeurons nOutputNeurons],
		{ "logsig", "tansig", "purelin"},
		"traingd",
		"learngdm",
		"mse"
	);

% stały seed wag początkowych, deiniowany lub wczytany

	IW = MLPnet.IW;
	LW = MLPnet.LW;
	B = MLPnet.b;

	iwFileName = sprintf("./initial_weights/%d_%d_%d_robot_iw.mat", nHiddenNeurons, n2HiddenNeurons, nOutputNeurons);
	iwExist = sprintf("exist('%s','file')", iwFileName);	
	iwLoad = sprintf("load '%s'", iwFileName);
	iwSave = sprintf("save '%s' IW", iwFileName);

	if(eval(iwExist) == 2)
	    eval(iwLoad);
	    MLPnet.IW = IW;		
	else
	    eval(iwSave);
	endif;

	lwFileName = sprintf("./initial_weights/%d_%d_%d_robot_lw.mat", nHiddenNeurons, n2HiddenNeurons, nOutputNeurons);
	lwExist = sprintf("exist('%s','file')", lwFileName);	
	lwLoad = sprintf("load '%s'", lwFileName);
	lwSave = sprintf("save '%s' LW", lwFileName);

	if(eval(lwExist) == 2)
	    eval(lwLoad);
	    MLPnet.LW = LW;		
	else
	    eval(lwSave);
	endif;

	bFileName = sprintf("./initial_weights/%d_%d_%d_robot_b.mat", nHiddenNeurons, n2HiddenNeurons, nOutputNeurons);
	bExist = sprintf("exist('%s','file')", bFileName);	
	bLoad = sprintf("load '%s'", bFileName);
	bSave = sprintf("save '%s' B", bFileName);

	if(eval(bExist) == 2)
	    eval(bLoad);
	    MLPnet.b = B;
	else
	    eval(bSave);
	endif;

% Zapis sieci

	%saveMLPStruct(MLPnet,"MLP_untrained.txt");

% define validation data new, for MATLAB(TM) compatibility

	VV.P = mValiInput;
	VV.T = mValliOutput;
	
% Normalizacja

	VV.P = trastd(VV.P,cMeanInput,cStdInput);

%[net,tr,out,E] = train(MLPnet,mInputN,mOutput,[],[],VV);

	MLPnet.trainParam.show = 1;
	[net] = train(MLPnet,mTrainInputN,mTrainOutput,[],[],VV);

%Zapisaujemy wytrenowaną sieć

%	saveMLPStruct(net,"MLP_trained.txt");


% Normalizacja zbioru testowego wg(prestd(mTrainInput))

	[mTestInputN] = trastd(mTestInput,cMeanInput,cStdInput);

%testujemy
	[simOut] = sim(net,mTestInputN); #%,Pi,Ai,mTestOutput);
	

%wynik
	disp("Out: ROBOT NAV");
	accLevel = 0.6;%0.6;
	rejLevel = 0.4;%0.3;
	errorCount = 0;
	% odpowiedź
	simOut = simOut';

	%oczekiwana
	mTestOutput = mTestOutput';


	%[mTestOutput simOut]
	%break;
	
	for idx = 1:rows(simOut)
		hi = 0;
		lo = 0;
		for idy = 1:length(simOut(idx, :))
			if(mTestOutput(idx, idy) == 1)
				if(simOut(idx, idy) > accLevel)
					hi = 1;
				endif;
			else
				if(simOut(idx, idy) < rejLevel)
					lo = lo +1;
				endif;				
			endif;	
		endfor;
		if((hi + lo) != numClasses)
			errorCount = errorCount +1;
			%[simOut(idx, :) mTestOutput(idx, :)]
		endif;
	endfor;
	%[mTestOutput simOut]
	idx
	errorRatio = errorCount/idx;
	disp("Out: procent błędnych rozpoznań");
	errorRatio
