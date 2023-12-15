clear
clc

set(0,'DefaultLineLineWidth',2);

rng default;  %% for reproducibility

%% set dimensions before calling the generating function
inSz = 2;          % number of slow variables + 1
outSz = 1;         % number of fast variables
nICs = 10;         % number of IC per variable
fTrainTest = 2;    % 1 for Training/Validation Set, 2 for Testing Set
firstData = true;  % if true, the testing set is produced and accuracy of the GSPT analytic expressions is assessed
                   % if false, only the accuracy is assessed at previously constructed Testing Set
                   
%% set the parameters for Segel Slemrod form of MM ode model
KM = 1e-2;
K = 1e-1;
s0 = 1e-3;
c0 = 0.;
kappa = KM/s0;
sigma = K/s0;

%% Create Training/Validation Set
if fTrainTest == 1
    %% Create Train Set
    nSpecs = inSz;
    ns = 20;          % samples per trajectory
    %
    %% range of epsilon
    eps_stepsPO = 5;
    eps_start = -4;
    eps_end = -1;
    eps_grid = logspace(eps_start,eps_end,(eps_stepsPO-1)*(eps_end-eps_start)+1);
    eps_grid = sort(eps_grid);

    %% range of ICs
    rICs = [0 2; 1. 2.];  
    %
    yAll = [];
    tAll = [];
    for i=1:numel(eps_grid)
        % Create 10 random ICs
        yICs = rICs(1:2,1) + (rICs(1:2,2)-rICs(1:2,1)).*rand(2,nICs^(nSpecs-1));
        parVec = [kappa sigma eps_grid(i)];
        % run to get the trajectories
        ySel = [];
        tSel = [];
        for j = 1:nICs^(nSpecs-1)
            [yEps1, tSol] = getMMdata(parVec,0,yICs(:,j),false,0);    % no cutting for transient or outside Omega, inside function
            % cut in the domain Omega [1e-6, 1] here
            yCut = yEps1(1,:)>=1e-6;
            yEps1 = yEps1(:,yCut);
            tSol = tSol(1,yCut);
            %% select equidistant in time points
            t_grid = linspace(tSol(1,1),tSol(1,end),ns);
            t_idx = zeros(1,ns);
            for ij=1:ns
                [~,t_idx(1,ij)] = min(abs(t_grid(1,ij)-tSol));
            end 
            tSel = [tSel tSol(1,t_idx)];
            ySel = [ySel yEps1(:,t_idx)];
        end
        yAll = [yAll ySel];
        tAll = [tAll tSel];     
    end
    allData = yAll;
    save MMTrain allData;
    save MMTraint tAll;

elseif fTrainTest == 2
    %% Create SIM data for 1 epsilon only for visualization (Fig 1a)
    ns = 400;
    nSpecs = inSz;
    if firstData
        eps = 1e-2;
        DomC = [0 1];                   % domain cut for plots
        eps_grid = repmat(eps,1,nICs^(nSpecs-1));
        %% create the data
        rICs = [0 2; 2. 3.];  
        yICs = rICs(1:2,1) + (rICs(1:2,2)-rICs(1:2,1)).*rand(2,nICs^(nSpecs-1));
        yAll = [];
        tAll = [];
        for i=1:numel(eps_grid)
            parVec = [kappa sigma eps_grid(i)];
            [yEps1, tSol] = getMMdata(parVec,ns,yICs(:,i),true,DomC);
            yAll = [yAll yEps1];
            tAll = [tAll tSol];
        end
        allData = yAll;
        save MMTestEps1e-2 allData;
        save MMTestEps1e-2t tAll;
        save MMTestEps1e-2ICs yICs;
    end

    %% Create Testing Set
    ns = 100;   % number of samples per trajectory
    nSpecs = inSz;
    if firstData
        nTestEps = 50;   % number of epsilon values to test
        %% range of epsilon to test
        eps_start = -4;
        eps_end = -1;
        eps = 10.^(eps_end+(eps_start-eps_end)*rand(1,nTestEps));
        eps_grid = repmat(eps,1,nICs^(nSpecs-1));
        eps_grid = sort(eps_grid);
        
        DomC = [1e-6 1];   % domain Omega we are interested in
        %% range of initial conditions (out of Omega)
        rICs = [0 2; 2. 3.];  
        yICs = rICs(1:2,1) + (rICs(1:2,2)-rICs(1:2,1)).*rand(2,nTestEps*nICs^(nSpecs-1));
        %
        %% create the training data sets
        yAll = [];
        tAll = [];
        for i=1:numel(eps_grid)
            parVec = [kappa sigma eps_grid(i)];
            [yEps1, tSol] = getMMdata(parVec,ns,yICs(:,i),true,DomC);    % cut the transient and inside Omega, inside the function
            yAll = [yAll yEps1];
            tAll = [tAll tSol];     
        end
        allData = yAll;
        save MMTest allData;
        save MMTestt tAll;
        save MMTestICs yICs;
    else
        load MMTest allData;
        load MMTestt tAll;
        load MMTestICs yICs;
    end
    Xtest = allData(end-inSz+1:end,:);
    Ytest = allData(1,:);

    %% Test Set errors of SPT/GSPT expressions
    %% errors of SPT/GSPT/CSP
    % sQSSA
    sQSSA = Xtest(1,:).*(kappa+1)./(kappa+Xtest(1,:));   % sQSSA 
    % O(epsilon) SPT
    SPT_o1C = (kappa*(kappa+1).^2./(sigma*(kappa+Xtest(1,:)).^3)).*(2*sigma.*Xtest(1,:)./(kappa + Xtest(1,:))...
            - Xtest(1,:) + (Xtest(1,:).*(kappa-sigma)/kappa).*log((kappa+Xtest(1,:))./((kappa+1).*Xtest(1,:))));
    o1_SPT = sQSSA + Xtest(2,:).*SPT_o1C; % oe1 + eps*Higher Order Correction of SPT
    % O(epsilon) GSPT
    GSPT_o1C = (kappa*(kappa+1)^3.*Xtest(1,:))./((kappa+Xtest(1,:)).^4);
    o1_GSPT = sQSSA + Xtest(2,:).*GSPT_o1C; % oe1 + eps*Higher Order Correction of GSPT
    % O(epsilon^2) GSPT
    GSPT_o2C = -(kappa*(kappa+1)^5.*Xtest(1,:).*(kappa^2+3*sigma*Xtest(1,:)+kappa*(Xtest(1,:)-...
                2*sigma)))./(sigma*(kappa+Xtest(1,:)).^7);
    o2_GSPT = sQSSA + Xtest(2,:).*GSPT_o1C + Xtest(2,:).^2.*GSPT_o2C; % oe1 + eps*oe2+ eps^2*oe3;
    %
    % CSP one iteration
    CSPo2 = (sigma*(kappa + Xtest(1,:)).^2 + Xtest(2,:).*(1 + kappa)^2.*(kappa - sigma + 2*Xtest(1,:)) - ... 
            sqrt(Xtest(2,:).^2*(1 + kappa)^4*(kappa - sigma)^2 + sigma^2*(kappa + Xtest(1,:)).^4 + ...
            2*Xtest(2,:).*(1 + kappa)^2*sigma.*(kappa + Xtest(1,:)).*(kappa*(kappa - sigma) + ...
            (kappa + sigma).*Xtest(1,:))))./(2.*Xtest(2,:)*(1 + kappa).*(kappa - sigma + Xtest(1,:)));
    sQSSAMSE = mse(sQSSA,Ytest);
    sQSSALinf = norm(sQSSA-Ytest,Inf);
    sQSSAL2 = norm(sQSSA-Ytest,2);
    SPToe2MSE = mse(o1_SPT,Ytest);
    SPToe2Linf = norm(o1_SPT-Ytest,Inf);
    SPToe2L2 = norm(o1_SPT-Ytest,2);
    GSPToe2MSE = mse(o1_GSPT,Ytest);
    GSPToe2Linf = norm(o1_GSPT-Ytest,Inf);
    GSPToe2L2 = norm(o1_GSPT-Ytest,2);
    GSPToe3MSE = mse(o2_GSPT,Ytest);
    GSPToe3Linf = norm(o2_GSPT-Ytest,Inf);
    GSPToe3L2 = norm(o2_GSPT-Ytest,2);
    CSPoe2MSE = mse(CSPo2,Ytest);
    CSPoe2Linf = norm(CSPo2-Ytest,Inf);
    CSPoe2L2 = norm(CSPo2-Ytest,2);
    fprintf('Test Set errors: on data of SIM \n')
    fprintf('L2  :    sQSSA         SPT O(eps)       GSPT O(eps)        GSPT O(eps^2)      CSP O(eps)     \n');
    fprintf('       %e   %e     %e       %e     %e     \n',sQSSAL2,SPToe2L2,GSPToe2L2,GSPToe3L2,CSPoe2L2);
    fprintf('Linf:    \n');
    fprintf('       %e   %e     %e       %e     %e     \n',sQSSALinf,SPToe2Linf,GSPToe2Linf,GSPToe3Linf,CSPoe2Linf);
    fprintf('MSE :    \n');
    fprintf('       %e   %e     %e       %e     %e     \n',sQSSAMSE,SPToe2MSE,GSPToe2MSE,GSPToe3MSE,CSPoe2MSE);

end
