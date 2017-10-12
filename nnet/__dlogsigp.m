## Author: Artur Augustyniak

function a = __dlogsigp(n)
% Wg tadeusiewicz 1993 str 58 - 61 pochodna bez parametru b
% Stosuję Wg \cite[s. 39-47]{osowski1997sieci}
  global betaGlobParam;	
  a = betaGlobParam * n .* (1-n);
endfunction
