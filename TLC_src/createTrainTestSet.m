clear
clc

set(0,'DefaultLineLineWidth',2);

rng default;  %% for reproducibility

%% set dimensions before calling the generating function
inSz = 3;           % number of slow variables + 1
outSz = 1;          % number of fast variables
fTrainTest = 1;     % 1 for Training/Validation Set, 2 for Testing Set
firstData = false;   % if true, the testing set is produced and accuracy of the GSPT analytic expressions is assessed
                    % if false, only the accuracy is assessed at previously constructed Testing Set

%% set the parameters of the full model
a = 0.1;
b = 0.6;
k = 1;

%% Preprocess Step to get good training data: Finf the center of the LC as epsilon varies
%% find center of limit cycle
% syms x1 y1 z1;
% %% eps = 1e-4;
% f1 = y1^2*z1-k*x1*y1;
% f2 = a*z1+y1^2*z1-y1+eps*x1;
% f3 = -a*z1-y1^2*z1+b;
% assume(x1,'real');
% sol = vpasolve([f1==0,f2==0,f3==0],[x1,y1,z1],'Random',true);
% SS = [double(sol.x1) double(sol.y1) double(sol.z1)];
% LCcenter = SS(1,:);
% %% save LCcenter4 LCcenter;

%% Create Training/Validation Set
if fTrainTest == 1
    nSpecs = inSz;  
    nICs = 5;   % number of IC per variable
    ns = 20;    % number of samples per trajectory
    %% range of epsilon
    eps_stepsPO = 5;
    eps_start = -4;
    eps_end = -1;
    eps_grid = logspace(eps_start,eps_end,(eps_end-eps_start)*(eps_stepsPO-1)+1);
    eps_grid = sort(eps_grid,'ascend');
    
    %% Set Omega
    DomC = [0.1 1.4; 0.3 2.3];   
    %% range of ICs, close to the boundary of Omega (interior and exterior of LC)
    load LCcenter4 LCcenter; %% assume it here for every epsilon (it does not matter, I want just interior traj of LC)
    rICs = [0 2; 0.5 1.0; 0.8 1.6];  % these include the LC, so start outside or inside
    rICs2 = [0 2; 0.3 1.2; 0.4 2.0];
        
    yICs = [];
    yAll = [];
    tAll = [];
    %% for each epsilon, take a set of random ICs and construct trajectories
    for i=1:numel(eps_grid)
        %% ICs to start outside of LC
        x1ICs = rICs(1,1) + (rICs(1,2)-rICs(1,1)).*rand(1,nICs^(nSpecs-1));
        y1ICs = (rICs(2:3,2)-rICs2(2:3,1)).*(randi(2,2,0.8*nICs^(nSpecs-1))-1) + rICs2(2:3,1) +  (rICs(2:3,1)-rICs2(2:3,1)).*rand(2,0.8*nICs^(nSpecs-1));
        %% ICs to start inside of LC
        y2ICs = LCcenter(2:3)' -0.1 + 2*0.1*rand(2,0.2*nICs^(nSpecs-1));
        yICs1 = [x1ICs; [y1ICs y2ICs]];
        yICs = [yICs yICs1];
        %
        parVec = [a b k eps_grid(i)];
        %% get the trajecotries for all ICs
        ySel = [];
        tSel = [];
        for j=1:nICs^(nSpecs-1)
            [yEps1, tSol] = getTLCdata(parVec,nSpecs,ns,yICs(:,j),false,0);   % no cutting for transient, no cutting in Omega
            %% check if collocation points are on Omega
            [~,idxs] = find(yEps1(2,:)>DomC(1,1) & yEps1(2,:)<DomC(1,2) & yEps1(3,:)>DomC(2,1) & yEps1(3,:)<DomC(2,2));
            tSol = tSol(idxs);
            yEps1 = yEps1(:,idxs);
            %% take equidistant in time points!
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
    save TLCTrain allData;
    save TLCTraint tAll;

elseif fTrainTest == 2
    %% Create SIM data for 1 epsilon only for visualization (Fig 1c)
    ns = 400;       % number of samples per trajectory
    nICs = 5;       % number of IC per variable
    nSpecs = inSz;
    if firstData
        eps = 1e-2;
        load LCcenter2 LCcenter;
        DomC = [0.1 1.4; 0.3 2.3];                      % domain Omega
        eps_grid = repmat(eps,1,nICs^(nSpecs-1));
        %% create ICs to get the trajectories data
        rICs = [0 2; 0.2 1.3; 0.5 2.1];  % these include the LC, so start outside or inside
        % ICs to start outside of LC
        x1ICs = rICs(1,1) + (rICs(1,2)-rICs(1,1)).*rand(1,nICs^(nSpecs-1));
        y1ICs = rICs(2:3,2).*(randi(2,2,0.8*nICs^(nSpecs-1))-1) + rICs(2:3,1).*rand(2,0.8*nICs^(nSpecs-1));
        % ICs to start inside of LC
        y2ICs = LCcenter(2:3)' -0.1 + 2*0.1*rand(2,0.2*nICs^(nSpecs-1));
        yICs = [x1ICs; [y1ICs y2ICs]];
        %
        yAll = [];
        tAll = [];
        for i=1:numel(eps_grid)
            parVec = [a b k eps_grid(i)];
            [yEps1, tSol] = getTLCdata(parVec,nSpecs,ns,yICs(:,i),true,DomC);
            yAll = [yAll yEps1];
            tAll = [tAll tSol];
        end
        allData = yAll;
        save TLCTestEps1e-2 allData;
        save TLCTestEps1e-2t tAll;
        save TLCTestEps1e-2ICs yICs;
    end

    %% Create Testing Set
    nSpecs = inSz;
    ns = 100;       % number of samples per trajectory
    nICs = 5;       % for lesser data (too many)
    if firstData
        nTestEps = 50;      % number of epsilon values to test
        %% range of epsilon to test
        eps_start = -4;
        eps_end = -1;
        eps = 10.^(eps_end+(eps_start-eps_end)*rand(1,nTestEps));
        eps_grid = repmat(eps,1,nICs^(nSpecs-1));
        eps_grid = sort(eps_grid);       
        %% Set Omega
        DomC = [0.1 1.4; 0.3 2.3];   % domain cut for plots
        %% range of ICs, close to the boundary of Omega (interior and exterior of LC)
        load LCcenter4 LCcenter;        %% assume it here for every epsilon (it does not matter, I want just interior traj of LC)
        rICs = [0 2; 0.2 1.3; 0.5 2.1];  % these include the LC, so start outside or inside
        yICs = [];
        for i = 1:nTestEps
            %% ICs to start outside of LC
            x1ICs = rICs(1,1) + (rICs(1,2)-rICs(1,1)).*rand(1,nICs^(nSpecs-1));
            y1ICs = rICs(2:3,2).*(randi(2,2,0.8*nICs^(nSpecs-1))-1) + rICs(2:3,1).*rand(2,0.8*nICs^(nSpecs-1));
            %% ICs to start inside of LC
            y2ICs = LCcenter(2:3)' -0.1 + 2*0.1*rand(2,0.2*nICs^(nSpecs-1));
            yICs1 = [x1ICs; [y1ICs y2ICs]];
            yICs = [yICs yICs1];
        end
        %
        yAll = [];
        tAll = [];
        for i=1:numel(eps_grid)
            parVec = [a b k eps_grid(i)];
            [yEps1, tSol] = getTLCdata(parVec,nSpecs,ns,yICs(:,i),true,DomC);
            yAll = [yAll yEps1];
            tAll = [tAll tSol];      
        end
        allData = yAll;
        save TLCTest allData;
        save TLCTestt tAll;
        save TLCTestICs yICs;
    else
        load TLCTest allData;
        load TLCTestt tAll;
        load TLCTestICs yICs;
    end
    Xtest = allData(end-inSz+1:end,:);
    Ytest = allData(1,:);

    %% Test Set errors of GSPT/CSP expressions 
    % sQSSA
    sQSSA = Xtest(1,:).*Xtest(2,:)/k;   % sQSSA
    % O(epsilon) GSPT
    GSPT_o1C = (-(b*Xtest(1,:)) + Xtest(2,:).*(Xtest(1,:).*(1 + a + Xtest(1,:).^2) - ...
            (a + Xtest(1,:).^2).*Xtest(2,:)))./(k^2*Xtest(1,:));
    o1_GSPT = sQSSA + Xtest(3,:).*GSPT_o1C; % oe1 + eps*Higher Order Correction of GSPT
    % O(epsilon^2) GSPT
    GSPT_o2C = (b*Xtest(1,:).*(-(Xtest(1,:).*(1 + a + Xtest(1,:).^2)) + 2*(a + Xtest(1,:).^2).*Xtest(2,:)) + ... 
               Xtest(2,:).*(a*Xtest(1,:).*(Xtest(1,:) + 2*Xtest(1,:).^3 + Xtest(2,:) - 6*Xtest(1,:).^2.*Xtest(2,:)) + ... 
               a^2*(Xtest(1,:).^2 - 2*Xtest(1,:).*Xtest(2,:) - Xtest(2,:).^2) + ...
               Xtest(1,:).^3.*(-2*Xtest(2,:) + Xtest(1,:).*(3 + Xtest(1,:).^2 - 4*Xtest(1,:).*Xtest(2,:) + ...
               Xtest(2,:).^2))))./(k^3*Xtest(1,:).^3);
    o2_GSPT = sQSSA + Xtest(3,:).*GSPT_o1C + Xtest(3,:).^2.*GSPT_o2C;
    % CSP with one iteration
    CSPo2 =  ((Xtest(3,:)*k.*Xtest(1,:) + k^2*Xtest(1,:).^2 - a*Xtest(3,:)*k.*Xtest(2,:) + ...
            2*Xtest(3,:).^2.*Xtest(1,:).*Xtest(2,:) - Xtest(3,:)*k.*Xtest(1,:).^2.*Xtest(2,:)).*(1 - ...
            sqrt(1 - (4*Xtest(3,:).^2*k.*(-(b*Xtest(3,:).*Xtest(1,:).^2) + 2*Xtest(3,:).*Xtest(1,:).^2.*Xtest(2,:) + ...
            a*Xtest(3,:).*Xtest(1,:).^2.*Xtest(2,:) + k*Xtest(1,:).^3.*Xtest(2,:) + Xtest(3,:).*Xtest(1,:).^4.*Xtest(2,:) - ... 
            2*a*Xtest(3,:).*Xtest(1,:).*Xtest(2,:).^2 - 2*Xtest(3,:).*Xtest(1,:).^3.*Xtest(2,:).^2))./(Xtest(3,:)*k.*Xtest(1,:) + ...
            k^2.*Xtest(1,:).^2 - a*Xtest(3,:)*k.*Xtest(2,:) + 2*Xtest(3,:).^2.*Xtest(1,:).*Xtest(2,:) - ...
            Xtest(3,:)*k.*Xtest(1,:).^2.*Xtest(2,:)).^2)))./(2.*Xtest(3,:).^2*k);
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

