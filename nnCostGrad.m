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
## @deftypefn {} {@var{retval} =} nnCostGrad (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: SMDEEPAK <SMDEEPAK@CREATOR-AT-CLOU>
## Created: 2019-12-12

function [J, grad] = nnCostGrad (Theta, X, Y)
  
  X = [ones(size(X,1),1) X]; % Dimension = 4999x401
  
  m = size(X,1);

  Theta1 = reshape(Theta(1:10025), 25,401); % Dimension = 25x401
  Theta2 = reshape(Theta(10026:end), 10, 26); % Dimension = 26x10
  
  % Forward Propagation
  a_1 = X;
  z_2 = a_1 * Theta1'; % Dimension = 4999x25
  a_2 = [ones(size(X,1),1) sigmoid(z_2)]; % Dimension = 4999x26
  
  z_3 = a_2 * Theta2'; % Dimension = 4999x10
  a_3 = sigmoid(z_3);

  modified_y = zeros(size(Y,1), 10);
  for i=1:size(Y,1),
    modified_y(i,Y(i)) = 1;
  endfor
  
 % regularize_term = (1/(2*m)) *  ( sum(sum(Theta1(:,2:end).^2,2),1) + sum(sum(Theta2(:,2:end).^2,2),1) );
  J = (1/m) * sum(sum(- ( modified_y .* log(a_3) + (1-modified_y) .* log(1-a_3) ),2),1);
 % + regularize_term;
  
  %J = (1/m) * sum(sum(- ( modified_y .* log(a_3) + (1-modified_y) .* log(1-a_3) ),2),1)
  
  % Back Propagation
  
    delta1 = 0;
    delta2 = 0;
  
    d3 = a_3 - modified_y; % Dimension = 4999x10
    d2 = (d3 * Theta2)(:, 2:end) .* sigmoidGradient(z_2); % (4999x10 * 26x10')(:, 2:end) .* 4999x25 => 4999x25 .* 4999x25 = 4999x25
    
    delta1 = delta1 + (d2' * a_1); % Dimension = delta1 + (4999x25' * 4999x401) = 25x401 
    delta2 = delta2 + (d3' * a_2); % Dimension = delta2 + (4999x10' * 4999x26) = 10x26
        size(delta2)

    Theta1_grad = (1/m) * delta1; % Dimension = 25x401
    Theta2_grad = (1/m) * delta2; % Dimension = 10x26
        size(Theta2_grad)

    %regularized_Theta1 = ((1/m) * Theta1);
    %regularized_Theta2 = ((1/m) * Theta2);
            %size(regularized_Theta2)

    %Theta1_grad(:,2:end) = Theta1_grad(:,2:end);
   % + regularized_Theta1(:,2:end);
    %Theta2_grad(:,2:end) = Theta2_grad(:,2:end);
   % + regularized_Theta2(:,2:end);
                                   
grad = [Theta1_grad(:); Theta2_grad(:)] ;    

endfunction
