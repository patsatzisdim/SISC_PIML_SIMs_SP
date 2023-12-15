function [outData, tSol] = getTMDDdata(parVec,nSpecs,nPpT,y_init,f_SIM,DomC)
% getTMDDdata function to generate MM requested data points from random initial conditions for given epsilon. 
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
k1 = parVec(1);
k2 = parVec(2);
k3 = parVec(3);
k4 = parVec(4);
epsilon = parVec(5);

outData = [];
tSol = [];

%% for training/validation data sets
if ~f_SIM   
    %% integrate with events to cut in Omega
    tend = 100;
    tspan = [0 tend];
    opts = odeset('RelTol',1e-12,'AbsTol',1e-16,'Events',@(t,y) defineZero(t,y));

    [t, y, te, ye, ie] = ode15s(@(t,y) TMDDodeSPA(t,y,epsilon,k1,k2,k3,k4),tspan,y_init,opts);
    
    %% get the solution until the slow subsystem is valid, i.e., when epsilon/y<5 epsilon < 1
    % here, cut for domain Omega
    [~,t_idx] = min(abs(t-min(te(ie==4))));
    t = t(1:t_idx,1);
    y = y(1:t_idx,:);

    %% keep data
    outData = [outData y'];
    tSol = [tSol t'];

%% for testing data sets
else
    %% integrate with events
    tend = 100;
    tspan = [0 tend]; 
    rfFac = 1;
    opts = odeset('RelTol',1e-12,'AbsTol',1e-16,'Refine',rfFac,'Events',@(t,y) defineZero(t,y));
    
    [t, y, te, ye, ie] = ode15s(@(t,y) TMDDodeSPA(t,y,epsilon,k1,k2,k3,k4),tspan,y_init,opts);
    
    %% get the solution until the slow subsystem is valid, i.e., when epsilon/y<5 epsilon < 1
    % here, cut for domain Omega
    [~,t_idx] = min(abs(t-te(ie==4)));
    t = t(1:t_idx,1);
    y = y(1:t_idx,:);

    %% cut the transient before 5*O(epsilon) to be on SIM
    trCut = t>5*epsilon;
    t = t(trCut);
    y = y(trCut,:);

    %% cut in Omega range DomC (low boundary was cut above)
    y_idx = find(y(:,2)<=DomC(1,2),1,'first');
    y = y(y_idx:end,:);
    t = t(y_idx:end,:);

    %% In case you need larger grid, play with the integrator
    while size(y,1)<nPpT 
        rfFac = rfFac*2;
        opts.Refine = rfFac;
        [t, y, te, ye, ie] = ode15s(@(t,y) TMDDodeSPA(t,y,epsilon,k1,k2,k3,k4),tspan,y_init,opts);
        %% get the solution until the slow subsystem is valid, i.e., when epsilon/y<5 epsilon < 1
        [~,t_idx] = min(abs(t-te(ie==4)));
        t = t(1:t_idx,1);
        y = y(1:t_idx,:);
        %% cut the transient before 5*O(epsilon) to be on SIM
        trCut = t>5*epsilon;
        t = t(trCut);
        y = y(trCut,:);
        %% cut in Omega range DomC (low boundary was cut above)
        y_idx = find(y(:,2)<=DomC(1,2),1,'first');
        y = y(y_idx:end,:);
        t = t(y_idx:end,:);
    end


    %% now the trajectory is in desired range and with more than nPpT samples
    % select only nPpT of them to grid
    y_grid = linspace(min(y(:,2)),max(y(:,2)),nPpT);                       % equidistant in y here
    y_keep = zeros(1,nPpT);
    ytemp = y(:,2);
    for i=1:nPpT
        [~,temp] = min(abs(y_grid(1,i)-ytemp));
        if ismember(temp,y_keep)
            ytemp(temp,:) = [];
            [~,temp] = min(abs(y_grid(1,i)-ytemp));
        end
        y_keep(1,i) = temp;
    end
    y = y(y_keep,:);
    t = t(y_keep,:);
    
    %% output data
    outData = zeros(nSpecs+1,size(y,1));
    outData(1:nSpecs,:) = y'; 
    outData(end,:) = epsilon*ones(1,size(y,1));
    tSol = t';

end

end


%% event function to terminate integration when R changes order of magnitude
function [value, isterminal, direction] = defineZero(t,y)
    value = [y(1)-1e-5; y(2)-1e-5; y(3)-1e-5;  0.2-y(2)];    % the last is to not get out of Omega    
    isterminal = [1; 1; 1; 0];                               % stop integration in ALL events
    direction = [0; 0; -1; 0];                               % meeting event for either increseing or decresing values                                              
end

