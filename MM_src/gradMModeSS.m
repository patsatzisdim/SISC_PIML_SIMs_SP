%% Segel-Slemrod non-dimensional form
function [Jac_x, Jac_y] = gradMModeSS(t,y,epsilon,kappa,sigma)
    nVar = size(y,1);
    nPoints = size(y,2);
    Jac_x = zeros(nVar,nPoints);
    Jac_y = zeros(nVar,nPoints);

    Jac_x(1,:) = -(1./epsilon).*(kappa+y(2,:))/(kappa+1); % grad dydt(1) over x=y(1) 
    Jac_y(1,:) = (1./epsilon).*(1-y(1,:)/(kappa+1)); % grad dydt(1) over y=y(2)
    
    Jac_x(2,:) = (1/sigma)*(kappa-sigma+y(2,:)); % grad dydt(2) over x=y(1)
    Jac_y(2,:) = (1/sigma)*(-(kappa+1)+y(1,:)); % grad dydt(2) over y=y(2)

end