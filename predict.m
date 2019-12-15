## Copyright (C) 2019 SMDEEPAK
## 
## This program is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see
## <https://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} predict (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: SMDEEPAK <SMDEEPAK@CREATOR-AT-CLOU>
## Created: 2019-12-14

function a_3 = predict (Theta, X)
  %m = size(X,1);
  X = [ones(size(X,1),1) X];
  Theta1 = reshape(Theta(1:10025), 25,401); % Dimension = 25x401
  Theta2 = reshape(Theta(10026:end), 10, 26); % Dimension = 26x10
  
  a_1 = X;
  z_2 = a_1 * Theta1'; % Dimension = 4999x25
  a_2 = [ones(size(X,1),1) sigmoid(z_2)]; % Dimension = 4999x26
  
  z_3 = a_2 * Theta2'; % Dimension = 4999x10
  sigmoid(z_3)
  [number, index] = max(sigmoid(z_3), [],2);
  a_3 = index;
endfunction
