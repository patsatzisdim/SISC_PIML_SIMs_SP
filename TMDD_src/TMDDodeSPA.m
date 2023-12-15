%% Non-dimensional form corresponding to P_2
function dydt = TMDDodeSPA(t,y,epsilon,k1,k2,k3,k4)
    nVar = size(y,1);
    nPoints = size(y,2);
    dydt = zeros(nVar,nPoints);
    dydt(1,:) = (1./epsilon).*(-y(1,:).*y(2,:)+k1*y(3,:)+1) - k2*y(1,:);       % dx/dt is dRdt norm/ed
    dydt(2,:) = k3*(-y(1,:).*y(2,:)+k1*y(3,:))-k4*y(2,:);                    % dy/dt is dLdt norm/ed
    dydt(3,:) = k2*(y(1,:).*y(2,:)-k1*y(3,:))-y(3,:);                       % dz/dt is dRLdt norm/ed

end

