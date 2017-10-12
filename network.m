function [simOut,  mTestOutput] =  network (label, dataFile, LayersNum, LayersFunc, minTeachCaseNum, LMepochs)

% dla dostosowania wg. VCDim n krotna prezentacja zbioru uczącego

	tmpData = mData = load(dataFile);
	
	if(minTeachCaseNum != 0)
		[nRows, nColumns] = size(mData);	

		trainSetsMutipl = ceil(minTeachCaseNum/floor(nRows/2));

		for i = 1:trainSetsMutipl
			tmpData = vertcat(tmpData, mData);
		endfor;	
	
		mData = tmpData;	
		[nRows, nColumns] = size(mData);

		% pseudoloswa kolejność powielonego zbioru	
		order = order_init = randperm(nRows);

		orderFileName = sprintf("./initial_weights/label_%s_order.mat", label);
		orderExist = sprintf("exist('%s','file')", orderFileName);	
		orderLoad = sprintf("load '%s'", orderFileName);
		orderSave = sprintf("save '%s' order_init", orderFileName);

		if(eval(orderExist) == 2)
		    eval(orderLoad);		
		    order = order_init;
		else
		    eval(orderSave);
		endif;
	
		mData(order,:) = mData;
	endif;

% + kolumny wygenerowane na podstawie identyfikatorów klas

	[nRows, nColumns] = size(mData);
	mInput = mData(:,1:end-1);
	mOutput = mData(:,end:end);
	
	numClasses = LayersNum(length(LayersNum));	
	
	vOutput = zeros(nRows, numClasses);
	
	vOutput_temp = zeros(numClasses, nRows);
	tempVect = 0:numClasses:(numClasses*nRows-1);
	vOutput_temp(tempVect'+mOutput)=1;
	vOutput = vOutput_temp';

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

% Minimalny zbiór uczący

	nTrainSets = floor(nRows/2);

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

	MLPnet = newff(
		mMinMaxElements,
		LayersNum,
		LayersFunc,
		"trainlm",
		"learngdm",
		"mse"
	);
	
% jeśli nie ma zapisanych wag początkowych zapisz do dalszych testów

	IW = MLPnet.IW;
	LW = MLPnet.LW;
	B = MLPnet.b;

	iwFileName = sprintf("./initial_weights/label_%s_iw.mat", label);
	iwExist = sprintf("exist('%s','file')", iwFileName);	
	iwLoad = sprintf("load '%s'", iwFileName);
	iwSave = sprintf("save '%s' IW", iwFileName);

	if(eval(iwExist) == 2)
	    eval(iwLoad);
	    MLPnet.IW = IW;		
	else
	    eval(iwSave);
	endif;

	lwFileName = sprintf("./initial_weights/label_%s_lw.mat", label);
	lwExist = sprintf("exist('%s','file')", lwFileName);	
	lwLoad = sprintf("load '%s'", lwFileName);
	lwSave = sprintf("save '%s' LW", lwFileName);

	if(eval(lwExist) == 2)
	    eval(lwLoad);
	    MLPnet.LW = LW;		
	else
	    eval(lwSave);
	endif;

	bFileName = sprintf("./initial_weights/label_%s_b.mat", label);
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
	%http://www.mathworks.com/help/nnet/ref/trainlm.html
	MLPnet.trainParam.show = 1;
	MLPnet.trainParam.epochs = LMepochs;
	[net] = train(MLPnet,mTrainInputN,mTrainOutput,[],[],VV);

%Zapisaujemy wytrenowaną sieć

%	saveMLPStruct(net,"MLP_trained.txt");

% Normalizacja zbioru testowego wg(prestd(mTrainInput))

	[mTestInputN] = trastd(mTestInput,cMeanInput,cStdInput);

%testujemy

	[simOut] = sim(net,mTestInputN); #%,Pi,Ai,mTestOutput);
	
%wynik
	% odpowiedź
	simOut = simOut';

	%oczekiwana
	mTestOutput = mTestOutput';

endfunction
