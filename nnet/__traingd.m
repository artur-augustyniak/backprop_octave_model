% Autor: Artur Augustyniak
% Klasyczny Back Propagation
function [net] = __traingd(net,Im,Pp,Tt,VV)

% Parametry procesu uczenia, tymczasowo z LM
  epochs   = net.trainParam.epochs;
  goal     = net.trainParam.goal;
  maxFail  = net.trainParam.max_fail;
  minGrad  = net.trainParam.min_grad;
  mu       = net.trainParam.mu;
  muInc    = net.trainParam.mu_inc;
  muDec    = net.trainParam.mu_dec;
  muMax    = net.trainParam.mu_max;
  show     = net.trainParam.show;
  time     = net.trainParam.time;


%Pp Train inputs set ->
%Tt Train outputs -> 

trainInputs = Pp{1,1}';
trainOutputs = Tt{3,1}';

[nExamples, nInputs] = size(trainInputs);
[nExamples, nOutputs] = size(trainOutputs);

% głównym ograniczeniem jest liczba epok
for bpEpochs = 0:nExamples

  O = neural-net-output(network, e)
  

  T = teacher output for e

bpEpochs




endfor;

%net

break;
endfunction;
