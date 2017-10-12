% Dodaję ścieżkę biblioteki 
% TODO
% niestety w tym momencie
% trzeba parametryzować funkcje zmiennymi globalnymi

	addpath("./nnet");
	addpath("./vendors");

############################################
# %TESTOWE ZESTAWY
############################################

% zestaw dla danych http://archive.ics.uci.edu/ml/datasets/Wall-Following+Robot+Navigation+Data
%mData = load("./datasets/shuffle_robot.data");
%numClasses = 4;	
% Konwersja wyjścia dla zmiennej nominalnej/klasyfikacji.
% 4 klasy
% -- Move-Forward	[1 0 0 0]
% -- Slight-Right-Turn	[0 1 0 0]
% -- Sharp-Right-Turn	[0 0 1 0]
% -- Slight-Left-Turn	[0 0 0 1]


% Dane http://archive.ics.uci.edu/ml/datasets/Cardiotocography	
%mData = load("./datasets/shuffle_cardio.data");
%numClasses = 3;	
% Konwersja wyjścia dla zmiennej nominalnej/klasyfikacji.
% 3 klasy
% NSP - fetal state class code 
% N=normal	[1 0 0]
% S=suspect	[0 1 0]
% P=pathologic	[0 0 1]


% Dane http://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)
%mData = load("./datasets/shuffle_sat.data");
%numClasses = 7;	
% Konwersja wyjścia dla zmiennej nominalnej/klasyfikacji.
% 7 klas
% 1 red soil				[1 0 0 0 0 0 0]
% 2 cotton crop				[0 1 0 0 0 0 0]
% 3 grey soil				[0 0 1 0 0 0 0]
% 4 damp grey soil			[0 0 0 1 0 0 0]
% 5 soil with vegetation stubble	[0 0 0 0 1 0 0]
% 6 mixture class (all types present)	[0 0 0 0 0 1 0]
% 7 very damp grey soil			[0 0 0 0 0 0 1]


% Architektura sieci wg Kołmogorowa
% Przy założeniu warstwy wyjściowej sigm
% \cite[s. 94-95]{osowski1997sieci} 1 warstwa ukrytna (2N+1) neuronów
% w ogólnym przypadku (nieciągłe x -> y) dwie warstwy ukryte

% Wg \cite[s. 97-98]{osowski1997sieci} K~sqrt(N*M) ilośc neuronów ukrytych 
% może być przybliżona średnią geom. wymiarów we/wy

% Minimalny zbiór uczący wg. oszacowania VCDim \cite[s. 95-96]{osowski1997sieci}
% Jeśli sigmoidalna funkcja transferu sigm VCDim ~ 2*W(calasiec)

############################################
# %GLOBALNE PASKUDZTWA
############################################

	global label;	
	global betaGlobParam;

############################################
# %PARAMETRYZACJA
############################################

	%file = "./datasets/shuffle_robot.data";
	%attrNum = 24;
	%outNum = 4;	
	%label = 'ROBOT_24_6_3_4_input_logsigp_logsig_purelin';

	%file = "./datasets/shuffle_cardio.data";
	%attrNum = 36;
	%outNum = 3;	
	%label = 'CARDIOGRAPHY_36_6_3_3_input_logsigp_logsig_purelin';


	file = "./datasets/shuffle_sat.data";
	attrNum = 36;
	outNum = 7;	
	label = 'SATELITE_36_6_3_7_input_logsigp_logsig_purelin';

	LayersNum = [6, 3, outNum];
	LayersFunc = {"logsigp", "logsig", "purelin"};

	numIters = 10; #50;
	betaGlobParam = 0.04;
	betaStep = 0.3;
	LMepochs = 100;
	acceptLevel = 0.5;
	rejectLevel = 0.5;


############################################
# %START
############################################

	VCDim = 2 * attrNum * prod(LayersNum);
	minTeachCaseNum = 0;#10 * VCDim;
	t = vRF1 = vRF2 = vRF3 = vRF4 = vTRF = zeros(1, numIters);	

	for i = 1:numIters
		[simOut, mTestOutput] = network(label, file, LayersNum, LayersFunc, minTeachCaseNum, LMepochs);
		% plot dynamiki uczenia
			print(sprintf("./out/learning_mse_%s_beta_%f_iteration_%d.png",label,betaGlobParam, i));
			hold off;
			clf();
		% analiza generalizacji/ test danych - Macierz błędów
			[errMat, RF, TP, TN, FP, FN, TRF] = analyzer(label, acceptLevel, rejectLevel, simOut, mTestOutput, LayersNum, outNum);
			TRF
			errMat
		% zapis do tex'a
		save2tex (errMat, sprintf("./out/err_matrix_%s_beta_%f_iteration_%d.tex",label,betaGlobParam, i), [], [], 0, "l", "full");
		save2tex ([RF TP TN FP FN], sprintf("./out/params_matrix_%s_beta_%f_iteration_%d.tex",label,betaGlobParam, i), [], [], 0, "l", "full");
		% dane zbiorcze dla końcowego wykresu
			vRF1(i) = RF(1);
			vRF2(i) = RF(2);
			vRF3(i) = RF(3);
			vRF4(i) = RF(4);
			vRF5(i) = RF(5);
			vRF6(i) = RF(6);
			vRF7(i) = RF(7);
			vTRF(i) = TRF;
			t(i) = betaGlobParam;
		% koniec param ++		
		betaGlobParam = betaGlobParam + betaStep;
	endfor;	

 	p = plot(
	   	t,vTRF,"-b;TRF charakterystyka Klasyfikatora;",
		t,vRF1,"-r;RF charak. dla klasy -- Move-Forward;",
		t,vRF2,"-m;RF charak. dla klasy -- Slight-Right-Turn;",
		t,vRF3,"-g;RF charak. dla klasy -- Sharp-Right-Turn;",
		t,vRF4,"-c;RF charak. dla klasy -- Slight-Left-Turn;",
		t,vRF5,"-c;RF charak. dla klasy -- Slight-Left-Turn;",
		t,vRF6,"-c;RF charak. dla klasy -- Slight-Left-Turn;",
		t,vRF7,"-c;RF charak. dla klasy -- Slight-Left-Turn;"
	);
	set (p(1), "linewidth", 2)
 	axis( [0 betaGlobParam - betaStep 0 1] );
	t = title(label);
	xlabel("\\beta \\warstwa \\ukryta");
	ylabel('Rozpoznanie [%]');
	grid on;
	legend('location','southeast'); 
	set (t, 'interpreter', 'none')
	set (gca,'fontsize',10);
	print(sprintf("./out/total_rf_%s.png",label));


