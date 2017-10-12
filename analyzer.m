% Autor: Artur Augustyniak
% technika odrzuceń mikrut2010sieci s 70
% error count całego klasyfikatora
% ROC dla każdego wyjściowego neuronu
function [errMat, RF, TP, TN, FP, FN, TRF] = analyzer(label, accLevel, rejLevel, simOut, mTestOutput, LayersNum, numClasses)

errorCount = 0;
[nRows, nColumns] = size(simOut);
TP = TN = FP = FN = zeros(1, nColumns);

% maciez blędow  \cite[s. 84-85]{mikrut2010sieci}	

errMat = zeros(numClasses, numClasses);

% MAIN
for idx = 1:rows(simOut)

%klasa teoretyczna	

	[ThiVal, ThiIdx] = max(mTestOutput(idx, :));

%klasa praktyczna

	[PhiVal, PhiIdx] = max(simOut(idx, :));

% sprawdzenie progów wynik całego klasyfikatora dla TRF
% statystyka poszczególnych wyjść vTP vTN vFP vFN dla RF 

	tmpT = mTestOutput(idx, :);
	tmpT(ThiIdx) = [];
	tmpP = simOut(idx, :);
	tmpP(PhiIdx) = [];
	rejectionVector = tmpT .+ tmpP;
	[rvVal, rvIdx] = max(rejectionVector);
	if((ThiIdx != PhiIdx || PhiVal < accLevel) || (rvVal > rejLevel))
		errorCount = errorCount +1;
		normalTmpP = simOut(idx, :);
		%"normalizacja" wektora klasy praktycznej
		for idy = 1:length(normalTmpP)
			if(normalTmpP(idy) > accLevel)
			   normalTmpP(idy) = ceil(normalTmpP(idy));
			else
			   normalTmpP(idy) = floor(abs(normalTmpP(idy)));
			endif;
		endfor;

		TP = TP .+ (mTestOutput(idx, :) & normalTmpP);
		TN = TN .+ (!(mTestOutput(idx, :) | normalTmpP));
		xorFalseVector = xor(mTestOutput(idx, :), normalTmpP);
		FP = FP .+ (xor(xorFalseVector, mTestOutput(idx, :)));
		FN = FN .+ (xor(xorFalseVector, normalTmpP));
	else
	      TN =  1 .+ TN;
	      TP(PhiIdx) = TP(PhiIdx)+ 1;
	      TN(PhiIdx) = TN(PhiIdx)-1;
	endif;
% maciez bledow  \cite[s. 84-85]{mikrut2010sieci}	
	errMat(PhiIdx, ThiIdx) = errMat(PhiIdx, ThiIdx) + 1;
endfor;
	% RF - współczynniki rozpoznania dla neuronów wyjściowych
	% TRF - sprawnośc klasyfikatora	
	RF = (TP + TN) ./ idx;
	TRF = 1 - errorCount/idx;
endfunction;
