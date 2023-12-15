%% Segel-Slemrod non-dimensional form
function dydt = MModeSS(t,y,epsilon,kappa,sigma)
    nVar = size(y,1);
    nPoints = size(y,2);
    dydt = zeros(nVar,nPoints);
    dydt(1,:) = (1./epsilon).*(y(2,:)-y(1,:).*(kappa+y(2,:))/(kappa+1)); 
    dydt(2,:) = (1/sigma)*(-(kappa+1)*y(2,:)+(kappa-sigma+y(2,:)).*y(1,:));
end

