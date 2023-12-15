%% Toy LC ode system
function dydt = ToyLCode_SP2(t,y,epsilon,a,b,k)
    nVar = size(y,1);
    nPoints = size(y,2);
    dydt = zeros(nVar,nPoints);
    dydt(1,:) = (1./epsilon).*(y(2,:).^2.*y(3,:)-k*y(1,:).*y(2,:));        % dx/dt 
    dydt(2,:) = a*y(3,:)+y(2,:).^2.*y(3,:)-y(2,:)+epsilon.*y(1,:);         % dy/dt 
    dydt(3,:) = -a*y(3,:)-y(2,:).^2.*y(3,:)+b;                             % dz/dt 

end

