%% Non-dimensional form corresponding to P_2
% gradients are here
function [Jac_x, Jac_y] = gradTMDDodeSPA(t,y,epsilon,k1,k2,k3,k4)
    nVar = size(y,1);
    nPoints = size(y,2);
    Jac_x = zeros(nVar,nPoints);
    Jac_y = zeros(nVar,nPoints);
    Jac_z = zeros(nVar,nPoints);
    
    %equations
%     dydt(1,:) = (1/epsilon)*(-y(1,:)*y(2,:)+k1*y(3,:)+1) - k2*y(1,:);       % dx/dt is dRdt norm/ed
%     dydt(2,:) = k3*(-y(1,:)*y(2,:)+k1*y(3,:))-k4*y(2,:);                    % dy/dt is dLdt norm/ed
%     dydt(3,:) = k2*(y(1,:)*y(2,:)-k1*y(3,:))-y(3,:);                       % dz/dt is dRLdt norm/ed

    uu=ones(1,nPoints);
    Jac_x(1,:) = (1./epsilon).*(-y(2,:))-k2; % grad dydt(1) over x=y(1) 
    Jac_y(1,:) = (1./epsilon).*(-y(1,:)); % grad dydt(1) over y=y(2)
    Jac_z(1,:) = (1./epsilon).*(+k1*uu); % grad dydt(1) over z=y(3)
    
    Jac_x(2,:) = k3*(-y(2,:)); % grad dydt(2) over x=y(1)
    Jac_y(2,:) = k3*(-y(1,:))-k4; % grad dydt(2) over y=y(2)
    Jac_z(2,:) = k3*(+k1*uu); % grad dydt(2) over z=y(3)
    
    Jac_x(3,:) = k2*(y(2,:)); % grad dydt(3) over x=y(1)
    Jac_y(3,:) = k2*(y(1,:)); % grad dydt(3) over y=y(2)
    Jac_z(3,:) = k2*(-k1*uu)-1; % grad dydt(3) over z=y(3)
    
    Jac_y=[Jac_y,Jac_z];

end