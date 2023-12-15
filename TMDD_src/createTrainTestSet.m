clear
clc

set(0,'DefaultLineLineWidth',2);

rng default;  %% for reproducibility

%% set dimensions before calling the generating function
inSz = 3;           % number of slow variables + 1
outSz = 1;          % number of fast variables
fTrainTest = 2;     % 1 for Training/Validation Set, 2 for Testing Set
firstData = false;   % if true, the testing set is produced and accuracy of the GSPT analytic expressions is assessed
                    % if false, only the accuracy is assessed at previously constructed Testing Set

%% set the parameters for full model
kon = 0.091;
koff = 0.001;
kel = 0.0015;
ksyn = 0.11;
kdeg = 0.0089;
kint = 0.003;
L0 = 100;

% parameters of the SPA form
k1 = koff/kdeg;
k2 = kdeg/kint;
k3 = ksyn/(kint*L0);
k4 = kel/kint;

%% Create Training/Validation Set
if fTrainTest == 1 
    nSpecs = inSz;
    nICs = 5;          % number of IC per variable
    ns = 20;           % samples per trajectory
    %% range of epsilon
    eps_stepsPO = 5;
    eps_start = -4;
    eps_end = -1;
    eps_grid = logspace(eps_start,eps_end,(eps_end-eps_start)*(eps_stepsPO-1)+1);
    
    %% range of Initial conditions, outside Omega but close to the boundary
    % Omega = [0.2 2.0] x [1.3 2.9]
    rICs = [0 2; 2. 2.4; 1.3 2.3];  %% close to where I want to start it
    yICs = [rICs(1,1)+(rICs(1,2)-rICs(1,1))*rand(1,nICs^(nSpecs-1)); 
        rICs(2,1)+(rICs(2,2)-rICs(2,1))*rand(1,nICs^(nSpecs-1)) ; 
        rICs(3,1)+(rICs(2,2)-rICs(2,1))*rand(1,nICs^(nSpecs-1))];    
    yAll = [];
    tAll = [];
    %% for each epsilon and IC, construct trajectories
    for i=1:numel(eps_grid)
        parVec = [k1 k2 k3 k4 eps_grid(i)];
        %% run all ICs
        ySel = [];
        tSel = [];
        % for each IC loop
        for j=1:nICs^(nSpecs-1)
            [yEps1, tSol] = getTMDDdata(parVec,nSpecs,0,yICs(:,j),false,0);     % no cutting for transient, inside function
            % Omega is cut inside the function, with events function
            %% select equidistant in time points
            t_grid = linspace(tSol(1,1),tSol(1,end),ns);
            t_idx = zeros(1,ns);
            for ij=1:ns
                [~,t_idx(1,ij)] = min(abs(t_grid(1,ij)-tSol));
            end
            tSel = [tSel tSol(1,t_idx)];
            ySel = [ySel yEps1(:,t_idx)];         
        end
        yAll = [yAll [ySel; eps_grid(i)*ones(1,ns*nICs^(nSpecs-1))] ];
        tAll = [tAll tSel];
    end
    allData = yAll;
    save TMDDTrain allData;
    save TMDDTraint tAll;

elseif fTrainTest == 2
    %% Create SIM data for 1 epsilon only for visualization (Fig 1b)
    ns = 100;           % # samples
    nICs = 10;  
    nSpecs = inSz;
    if firstData
        eps = 1e-2;
        DomC = [0.2 2; 1.3 2.9];        % domain Omega 
        eps_grid = repmat(eps,1,nICs^(nSpecs-1));
        %% create ICs to get the trajectories data
        rICs = [0 2; 3 4; 0 1];   % range of initial conditions (outside Omega)
        yICs = rICs(:,1) + (rICs(:,2)-rICs(:,1)).*rand(nSpecs,nICs^(nSpecs-1));  
        %
        yAll = [];
        tAll = [];
        for i=1:numel(eps_grid)
            parVec = [k1 k2 k3 k4 eps_grid(i)];
            [yEps1, tSol] = getTMDDdata(parVec,nSpecs,ns,yICs(:,i),true,DomC);
            yAll = [yAll yEps1];
            tAll = [tAll tSol];
        end
        allData = yAll;
        save TMDDTestEps1e-2 allData;
        save TMDDTestEps1e-2t tAll;
        save TMDDTestEps1e-2ICs yICs;
    end
    
    %% Create Testing Set
    ns = 100;   % number of samples per trajectory
    nSpecs = inSz;
    nICs = 5;  
    if firstData
        nTestEps = 50;  % number of epsilon values to test
        %% range of epsilon to test
        eps_start = -4;
        eps_end = -1;
        eps = 10.^(eps_end+(eps_start-eps_end)*rand(1,nTestEps));
        eps_grid = repmat(eps,1,nICs^(nSpecs-1));
        eps_grid = sort(eps_grid,'descend');
        %% create the data
        DomC = [0.2 2; 1.3 2.9];        % domain Omega 
        %% range of initial conditions (out of Omega)
        rICs = [0 2; 3 4; 0 1];
        yICs = rICs(:,1) + (rICs(:,2)-rICs(:,1)).*rand(nSpecs,nTestEps*nICs^(nSpecs-1));
        yAll = [];
        tAll = [];
        for i=1:numel(eps_grid)
            parVec = [k1 k2 k3 k4 eps_grid(i)];
            [yEps1, tSol] = getTMDDdata(parVec,nSpecs,ns,yICs(:,i),true,DomC);
            yAll = [yAll yEps1];
            tAll = [tAll tSol];
        end
        allData = yAll;
        save TMDDTest allData;
        save TMDDTestt tAll;
        save TMDDTestICs yICs;
    else
        load TMDDTest allData;
        load TMDDTestt tAll;
        load TMDDTestICs yICs; 
    end
    Xtest = allData(end-inSz+1:end,:);
    Ytest = allData(1,:);

    %% Test Set errors of GSPT/CSP expressions 
    % sQSSA
    sQSSA = (k1*Xtest(2,:)+1)./Xtest(1,:);   % sQSSA
    % O(epsilon) GSPT
    GSPT_o1C = -((k3 + k1*k3*Xtest(2,:) + Xtest(1,:).*(k2 + k1*k2 + k4 + k1*(-1 +...
            k2 + k4).*Xtest(2,:)))./Xtest(1,:).^3);
    o1_GSPT = sQSSA + Xtest(3,:).*GSPT_o1C; % oe1 + eps*Higher Order Correction of GSPT
    % O(epsilon^2) GSPT
    GSPT_o2C = (k3^2*(1 + k1*Xtest(2,:)).*(4 + k1*Xtest(2,:)) + Xtest(1,:).^2.*((k2 + k4)*(k2 + 2*k4) + ...
               k1*k2*(-1 + 3*k2 + 4*k4) + k1*(-1 + k2 + k4)*(-1 + k2 + 2*k4)*Xtest(2,:) + ...
               k1^2*k2*(k2 + (-1 + k2 + k4)*Xtest(2,:))) + k3*Xtest(1,:).*(6*k4 + k1*Xtest(2,:).*(-4 + ...
               7*k4 + k1*(-1 + k4)*Xtest(2,:)) + k2*(4 + k1*(5 + Xtest(2,:).*(5 + k1*(2 + Xtest(2,:)))))))./Xtest(1,:).^5;
    o2_GSPT = sQSSA + Xtest(3,:).*GSPT_o1C + Xtest(3,:).^2.*GSPT_o2C;
    % CSP one iteration
    CSPo2 = -(Xtest(3,:).^2*k2^2 + Xtest(3,:).*((2 + k1)*k2 + k4).*Xtest(1,:) + Xtest(1,:).^2 - ...
             Xtest(3,:)*k1*k3.*Xtest(2,:) - sqrt((Xtest(3,:).^2*k2^2 + Xtest(3,:).*((2 + k1)*k2 + ...
             k4).*Xtest(1,:) + Xtest(1,:).^2 - Xtest(3,:)*k1*k3.*Xtest(2,:)).^2 + ...
             4*Xtest(3,:)*k3.*Xtest(1,:).*(Xtest(3,:)*k2 + Xtest(1,:) + k1*(Xtest(3,:)*(1 + ...
             k2 + k1*k2) + Xtest(1,:)).*Xtest(2,:))))./(2.*Xtest(3,:)*k3.*Xtest(1,:));
    sQSSAMSE = mse(sQSSA,Ytest);
    sQSSALinf = norm(sQSSA-Ytest,Inf);
    sQSSAL2 = norm(sQSSA-Ytest,2);
    GSPToe2MSE = mse(o1_GSPT,Ytest);
    GSPToe2Linf = norm(o1_GSPT-Ytest,Inf);
    GSPToe2L2 = norm(o1_GSPT-Ytest,2);
    GSPToe3MSE = mse(o2_GSPT,Ytest);
    GSPToe3Linf = norm(o2_GSPT-Ytest,Inf);
    GSPToe3L2 = norm(o2_GSPT-Ytest,2);
    CSPoe2MSE = mse(CSPo2,Ytest);
    CSPoe2Linf = norm(CSPo2-Ytest,Inf);
    CSPoe2L2 = norm(CSPo2-Ytest,2);
    fprintf('L2  :    sQSSA       GSPT O(eps)       GSPT O(eps^2)        CSP O(eps)     \n');
    fprintf('       %e   %e     %e     %e       \n',sQSSAL2,GSPToe2L2,GSPToe3L2,CSPoe2L2);
    fprintf('Linf:    \n');
    fprintf('       %e   %e     %e     %e       \n',sQSSALinf,GSPToe2Linf,GSPToe3Linf,CSPoe2Linf);
    fprintf('MSE :    \n');
    fprintf('       %e   %e     %e     %e       \n',sQSSAMSE,GSPToe2MSE,GSPToe3MSE,CSPoe2MSE);
    

end

