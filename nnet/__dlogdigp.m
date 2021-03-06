## Copyright (C) 2007 Michel D. Schmid <michaelschmid@users.sourceforge.net>
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

## -*- texinfo -*-
## @deftypefn {Function File} {}[@var{a} = __dlogsig (@var{n})
##
## @end deftypefn

## @seealso{__dpurelin,__dtansig}

## Author: Michel D. Schmid


function a = __dlogsigp(n)

% wg ossowski sieci str 39 - 47
% wg tadeusiewicz 1993 str 58 - 61 pochodna bez parametru b -> szybsiej wolniej kręci
  global betaGlobParam;	
  nc = 1;% betaGlobParam;
  a = (nc * n) .*(1-n );

endfunction
