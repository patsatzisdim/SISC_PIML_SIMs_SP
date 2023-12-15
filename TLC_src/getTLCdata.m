function [outData, tSol] = getTLCdata(parVec,nSpecs,nPpT,y_init,f_SIM,DomC)
% getTLCdata function to generate MM requested data points from random initial conditions for given epsilon. 
%
%  Inputs:   - parVec: [k1 k2 k3 k4 epsilon]: the first 4 are fixed, the last, epsilon, changes
%            - nSpecs: number of variables
%            - nPpT: number of point per Trajectory
%            - y_init: intial conditions
%            - f_SIM: when true cut for t<5 epsilon
%            - DomC: Domain Omega to cut the trajectory
%
%  Outputs:  - outData: vector of (N+1) x # of samples, the last row is epsilon  
%            - tSol: time when the points are recorded

%
a = parVec(1);
b = parVec(2);
k = parVec(3);
eps = parVec(4);

outData = [];
tSol = [];

%% for training/validation data sets
if ~f_SIM
    %% integrate with events to stop integration when solution cuts Poincare section in very small distances (every period)
    tend = 1000;
    tspan = [0 tend];
    opts = odeset('RelTol',1e-4,'AbsTol',1e-6,'Events',@(t,y) PoincarePlane(t,y));
    [t, y, te, ye, ie]  = ode15s(@(t,y) ToyLCode_SP2(t,y,eps,a,b,k),tspan,y_init,opts);
    % 
    %% find when derivative of Poincare section cut is very small (<5e-3);i.e., when almost at Limit Cycle
    difPP = -diff(ye(:,3));
    % when the solution begins inside the LC
    if difPP(1,1)<0
        [~, TLCpp] = min(abs(-difPP-5e-3));
        timeLC1 = te(TLCpp+1);
    else
        [~, TLCpp] = min(abs(difPP-5e-3));
        timeLC1 = te(TLCpp+1);
    end

    %% Now accurately integrate until before on Limit Cycle
    tend = timeLC1;
    tspan = [0 tend];
    opts.RelTol = 1e-12;
    opts.AbsTol = 1e-16;
    [t, y]  = ode15s(@(t,y) ToyLCode_SP2(t,y,eps,a,b,k),tspan,y_init,opts);
    
    %% keep data
    outData = [outData y'];
    outData = [outData; eps*ones(1,size(y,1))];
    tSol = [tSol t'];

%% for testing data sets
else
    %% integrate with events to stop integration when solution cuts Poincare section in very small distances (every period)
    tend = 1000;
    tspan = [0 tend];
    rfFac = 1;
    opts = odeset('RelTol',1e-4,'AbsTol',1e-6,'Refine',rfFac,'Events',@(t,y) PoincarePlane(t,y));
    [t, y, te, ye, ie]  = ode15s(@(t,y) ToyLCode_SP2(t,y,eps,a,b,k),tspan,y_init,opts);
    % 
    %% find when derivative of Poincare section cut is very small (<5e-3);i.e., when almost at Limit Cycle
    difPP = -diff(ye(:,3));
    % when the solution begins inside the LC
    if difPP(1,1)<0
        [~, TLCpp] = min(abs(-difPP-5e-3));
        timeLC1 = te(TLCpp+1);
    else
        [~, TLCpp] = min(abs(difPP-5e-3));
        timeLC1 = te(TLCpp+1);
    end

    %% Now accurately integrate until before on Limit Cycle
    tend = timeLC1;
    tspan = [0 tend];
    opts.RelTol = 1e-12;
    opts.AbsTol = 1e-16;
    [t, y]  = ode15s(@(t,y) ToyLCode_SP2(t,y,eps,a,b,k),tspan,y_init,opts);

    %% cut the transient before 30*O(epsilon) to be on SIM
    trCut = t>30*eps;
    t = t(trCut);
    y = y(trCut,:);

    %% cut inside Omega
    y_idx = find((y(:,2)<=DomC(1,1)) | (y(:,2)>=DomC(1,2)) | (y(:,3)<=DomC(2,1) | (y(:,3)>=DomC(2,2))),1,'last');
    if isempty(y_idx); y_idx = 1; end
    y = y(y_idx:end,:);
    t = t(y_idx:end,:);

    %% In case you need larger grid, play with the integrator
    while size(y,1)<nPpT 
        rfFac = rfFac*2;
        opts.Refine = rfFac;
        [t, y]  = ode15s(@(t,y) ToyLCode_SP2(t,y,eps,a,b,k),tspan,y_init,opts);
        %% cut the transient before 30*O(epsilon) to be on SIM
        trCut = t>30*eps;
        t = t(trCut);
        y = y(trCut,:);
        %% cut inside Omega
        y_idx = find((y(:,2)<=DomC(1,1)) | (y(:,2)>=DomC(1,2)) | (y(:,3)<=DomC(2,1) | (y(:,3)>=DomC(2,2))),1,'last');
        y = y(y_idx:end,:);
        t = t(y_idx:end,:);
    end


    %% now the trajectory is in desired range and with more than nPpT samples
    % select only nPpT of them to grid
    t_grid = linspace(t(1,1),timeLC1,nPpT);                                % equidistant in time here because it is oscillating
    t_keep = zeros(1,nPpT);
    ttemp = t;
    for i=1:nPpT
        [~,temp] = min(abs(t_grid(1,i)-ttemp));
        if ismember(temp,t_keep)
            ttemp(temp,:) = [];
            [~,temp] = min(abs(t_grid(1,i)-ttemp));
        end
        t_keep(1,i) = temp;
    end
    y = y(t_keep,:);
    t = t(t_keep,:);
    
    %% output data
    outData = zeros(nSpecs+1,size(y,1));
    outData(1:nSpecs,:) = y'; 
    outData(end,:) = eps*ones(1,size(y,1));
    tSol = t';

end


end


%% event function to record when y passes 0.7 (xz plane) only increasing
function [value, isterminal, direction] = PoincarePlane(t,y)
    value = y(2)-0.7;    % the last two are the CSP criteria for QSSA      
    isterminal = 0;                                               % stop integration in ALL events
    direction = 1;                          % meeting event for either increseing or decresing values
                                              
end


