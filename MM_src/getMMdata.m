function [outData, tSol] = getMMdata(parVec,nPpT,y_init,f_SIM,DomC)
% getMMsolSIM function to generate MM requested data points from random initial conditions for given epsilon. 
% The output points must lie on the SIM and after transient (say O(5*epsilon))
%
%  Inputs:   - parVec: [kappa sigma epsilon]
%            - nPpT: number of point per Trajectory
%            - y_init: intial conditions
%            - f_SIM: when true cut for t<5 epsilon
%            - DomC: Domain Omega to cut the trajectory
%
%  Outputs:  - outData: vector of (N+1) x # of samples, the last row is epsilon  
%            - tSol: time when the points are recorded

%
kappa = parVec(1);
sigma = parVec(2);
epsilon = parVec(3);

%% for training/validation data sets
if ~f_SIM
    tend = 100;
    tspan = [0 tend];
    rfFac = 1;
    endPoint = 1e-9;
    opts = odeset('RelTol',1e-12,'AbsTol',1e-16,'Refine',rfFac,'Events',@(t,y) defineZero(t,y,endPoint));
    [t, y, te, ye, ie] = ode15s(@(t,y) MModeSS(t,y,epsilon,kappa,sigma),tspan,y_init,opts);    

    y_Pos = y(:,1)>=0 & y(:,2)>=0;
    y = y(y_Pos,:);
    t = t(y_Pos,1);

    outData = [y'; epsilon*ones(1,size(y,1))]; 
    tSol = t';

else 
    tend = 100;
    tspan = [0 tend];
    rfFac = 1;
    endPoint = DomC(1);  % lower for y
    opts = odeset('RelTol',1e-12,'AbsTol',1e-16,'Refine',rfFac,'Events',@(t,y) defineZero(t,y,endPoint));
    [t, y, te, ye, ie] = ode15s(@(t,y) MModeSS(t,y,epsilon,kappa,sigma),tspan,y_init,opts);

    %% cut the transient before 5*O(epsilon) to be on SIM
    trCut = t>5*epsilon;
    t = t(trCut);
    y = y(trCut,:);
    
    %% cut in range Omega
    y_idx = find(y(:,2)<=DomC(1,2),1,'first');
    if y_idx == 1; error('Start higher on y(2)'); end
    if isempty(y_idx); y_idx = 1; end
    y = y(y_idx:end,:);
    t = t(y_idx:end,:);

    %% In case you need larger grid, play with the integrator
    while size(y,1)<nPpT 
        rfFac = rfFac*2;
        opts.Refine = rfFac;
        [t, y] = ode15s(@(t,y) MModeSS(t,y,epsilon,kappa,sigma),tspan,y_init,opts);
        trCut = t>5*epsilon;
        t = t(trCut);
        y = y(trCut,:);
        %% cut in range Omega
        y_idx = find(y(:,2)<=DomC(1,2),1,'first');
        if y_idx == 1; error('Start higher on y(2)'); end
        if isempty(y_idx); y_idx = 1; end
        y = y(y_idx:end,:);
        t = t(y_idx:end,:);
    end

    %% now the trajectory is in desired range and with more than nPpT samples
    % select only nPpT of them to grid
    y_grid = logspace(log10(min(y(:,2))),log10(max(y(:,2))),nPpT);  %% equidistant in y here
    y_grid = sort(y_grid,'descend');
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
    outData = [y'; epsilon*ones(1,size(y,1))]; 
    tSol = t';

end

end

%% event function to terminate integration when solution goes too low
function [value, isterminal, direction] = defineZero(t,y,endPoint)
    value = [y(1)-endPoint; y(2)-endPoint];                       % when going below endPoint is near zero     
    isterminal = [1; 0];                     % stop integration in ALL events
    direction = [-1; -1];                    % meeting event 1&2: when decreasing values of y 
end

