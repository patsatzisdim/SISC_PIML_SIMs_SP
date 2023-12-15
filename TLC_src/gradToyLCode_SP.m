function [Jac_x, Jac_y] = gradToyLCode_SP(t,y,epsilon,a,b,k)
    nVar = size(y,1);
    nPoints = size(y,2);
    Jac_x = zeros(nVar,nPoints);
    Jac_y = zeros(nVar,nPoints);
    Jac_z = zeros(nVar,nPoints);
%     dydt(1,:) = (1./epsilon).*(y(2,:).^2.*y(3,:)-k*y(1,:).*y(2,:));        % dx/dt 
%     dydt(2,:) = a*y(3,:)+y(2,:).^2.*y(3,:)-y(2,:)+epsilon.*y(1,:);         % dy/dt 
%     dydt(3,:) = -a*y(3,:)-y(2,:).^2.*y(3,:)+b;                             % dz/dt 
    Jac_x(1,:) = -(1./epsilon)*k.*y(2,:);
    Jac_x(2,:) = epsilon;
    Jac_x(3,:) = zeros(1,nPoints);

    Jac_y(1,:) = (1./epsilon).*(2*y(2,:).*y(3,:)-k*y(1,:));
    Jac_y(2,:) = 2*y(2,:).*y(3,:)-ones(1,nPoints);
    Jac_y(3,:) = -2*y(2,:).*y(3,:);

    Jac_z(1,:) = (1./epsilon).*y(2,:).^2;
    Jac_z(2,:) = a*ones(1,nPoints)+y(2,:).^2;
    Jac_z(3,:) = -a*ones(1,nPoints)-y(2,:).^2;

    Jac_y = [Jac_y, Jac_z];
end
