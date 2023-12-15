clear
clc

rng default; % for reproducibility

%% set dimensions before calling the generating function
inSz = 2;           % number of slow variables + 1
outSz = 1;          % number of fast variables
noiterations = 1;   % number of training runs to consider

%% Set Random Projections of RPNN
N = 81;      % number of neurons per hidden layer
fRPscale = 1;  

%% set the parameters for Segel Slemrod form of MM ode model
KM = 1e-2;
K = 1e-1;
s0 = 1e-3;
c0 = 0.;
kappa = KM/s0;
sigma = K/s0;

%% load training, test data
load MMTest allData;
Xtest = allData(end-inSz+1:end,:);
Ytest = allData(1,:);

load MMTrain allData;
Xtrain = allData(end-inSz+1:end,:);
Ytrain = allData(1,:);

%% Testing Set: accuracy of SPT/GSPT analytic expressions
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

%% Train Network to solve PIML problem
CPUrecs = zeros(noiterations,1);
trainMSE = zeros(noiterations,1);
trainLinf = zeros(noiterations,1);
trainL2 = zeros(noiterations,1);
CVMSE = zeros(noiterations,1);
CVLinf = zeros(noiterations,1);
CVL2 = zeros(noiterations,1);
testMSE = zeros(noiterations,1);
testLinf = zeros(noiterations,1);
testL2 = zeros(noiterations,1);
learnPar = zeros(noiterations,N*outSz+N*inSz+N);
Xdata = Xtrain;
Ydata = Ytrain;
for i = 1:noiterations
    %% Split to training and validation sets
    idx = randperm(size(Xdata,2),floor(size(Xdata,2)*0.2));
    XCV = Xdata(:,idx);
    YCV = Ydata(:,idx);
    Xtrain = Xdata;
    Ytrain = Ydata;
    Xtrain(:,idx) = [];
    Ytrain(:,idx) = [];
    %% fix internal weights and random initialize output layer 
    tic
    [Alpha, beta, Centr] = RPinternalWeights(N,Xtrain,fRPscale);        % get alphas and betas
    PHI= getRPmatrix(Alpha, beta, Xtrain);                              
    Wout = randn(outSz,N)/100;    % output weights    1xL

    %% Train RPNN
    fixedPar = [kappa sigma];
    learnables = Wout;
 
    %% Perform Newton with SVD
    maxIter = 30;               % max iterations allowed
    iter = 0;
    Fmin = funPIloss(learnables,Alpha,beta,Xtrain,fixedPar);    %% residuals for first iteration
    NFminP=norm(Fmin);          
    bestnorm=NFminP;  
    Cor = 1000;                 % for oscillating errors to be avoided
    contfail=0; maxcntf=3;      % allow three times
    while (iter <= maxIter) && (Cor-1>1e-3) && contfail<maxcntf
        iter=iter+1;
        % calculate the derivatives of residuals w.r.t. out weights
        dFdW = gradFminWout(learnables,Alpha,beta,Xtrain,fixedPar); 
        % solve the system dFdW * dW = - Fmin   with SVD
        dW = -Fmin*pinv(dFdW,1e-7);
        eta=dW;
        learnables = learnables + eta; 
        % calculate new residuals
        Fmin = funPIloss(learnables,Alpha,beta,Xtrain,fixedPar);
        newFminP=norm(Fmin);
        % check new error
        if newFminP<NFminP   
            Cor = NFminP/newFminP;
            NFminP = newFminP;
            contfail=0;
            if newFminP<bestnorm
                bestnorm=newFminP;
                bestlearnables=learnables;
            end
        else
            Cor=100;
            contfail=contfail+1;
            NFminP = newFminP;
        end
    end
    if iter>=maxIter; fprintf('Maximum number of NR iter reached with norm of residuals %f. \n',NFminP); end
    if contfail==maxcntf 
        fprintf('RPNN may not converged %f. \n',bestnorm); 
    else
        fprintf('RPNN CONVERGE %f. \n',NFminP);
    end
   
    %% keep best error solution (in case of oscillating errors)
    Wout = bestlearnables;
    CPUend = toc;
    CPUrecs(i,1) = CPUend;
    learnPar(i,:) = [Wout reshape(Alpha,1,[]) beta'];

    %% PINN train error and CV errors
    trainMSE(i,1) = mse(Fmin,zeros(size(Ytrain)));
    trainLinf(i,1) = norm(Fmin,Inf);
    trainL2(i,1) = norm(Fmin,2);
    CVres = funPIloss(Wout,Alpha,beta,XCV,fixedPar);
    CVMSE(i,1) = mse(CVres,zeros(size(YCV)));
    CVLinf(i,1) = norm(CVres,Inf);
    CVL2(i,1) = norm(CVres,2);

    %% Evaluation on test data (which on SIM)
    PHItest = getRPmatrix(Alpha, beta, Xtest);
    testSIM = Wout*PHItest;
    testMSE(i,1) = mse(testSIM,Ytest);
    testLinf(i,1) = norm(testSIM-Ytest,Inf);
    testL2(i,1) = norm(testSIM-Ytest,2);
end

%% keep best learned parameters w.r.t CV
[aisdjoasi,idx_bnet]=min(CVL2);
btestMSE = testMSE(idx_bnet,1);
btestLinf = testLinf(idx_bnet,1);
btestL2 = testL2(idx_bnet,1);
bestLearned = learnPar(idx_bnet,:);
save learned_PI_RPNN bestLearned;

%% Metrics on training
fprintf('-------PIML RPNN  TRAINING metrics --------\n')
fprintf('CPU times:    mean              min              max  \n');
fprintf('         %e      %e     %e     \n',mean(CPUrecs),min(CPUrecs),max(CPUrecs));
fprintf('Train err:     l2              lInf              MSE  \n');
fprintf('         %e      %e     %e     \n',mean(trainL2),mean(trainLinf),mean(trainMSE));
fprintf('CV    err:     l2              lInf              MSE  \n');
fprintf('         %e      %e     %e     \n',mean(CVL2),mean(CVLinf),mean(CVMSE));

%% Metrics on testing (on SIM data)
fprintf('-------PIML RPNN  TESTING metrics --------\n')
fprintf('Test  err:     l2              lInf              MSE  \n');
fprintf('         %e      %e     %e     \n',mean(testL2),mean(testLinf),mean(testMSE));
%% bestCV Metrics on testing (on SIM data)
fprintf('-------PIML PIRPNN best CV TESTING metrics --------\n')
fprintf('Test  err:     l2              lInf              MSE  \n');
fprintf('         %e      %e     %e     \n',btestL2,btestLinf,btestMSE);

return

%%% END OF MAIN ROUTINE


%% function to randomly determine the internal weights ACCORDING TO LOGISTIC SIGMOID activation 
function [Aa, bb, Centr] = RPinternalWeights(N,X,RPScale)
    Xmin=min(X,[],2);
    Xmax=max(X,[],2);
    DX=Xmax-Xmin;
    [d,np]=size(X);
    %% choose centers
    if RPScale == 1 % uniform scaling
        Centr=[DX(1:d-1)'.*rand(N,d-1)+Xmin(1:d-1)',...
            logspace(log10(Xmin(d))-1,log10(Xmax(d))+1,N)'];%(DX(d)+Xmin(d)/100)'.*exprnd(1,N,1)/5+Xmin(d)'-Xmin(d)/100];
        Aa = 2*(rand(N,d)-0.5);
    end
    %% Set bb so that inflection point being at the center
    bb= -sum(Aa.*Centr,2);
end

%% Physics-Informed Loss Function to be minimized for learning RPNN
% form f(h(y,e),y,e) - e*grad_y(h(y,e))*g(h(y,e),y,e)    where f = e*dx/dt and g = dy/dt 
%
% h(y,e) = PHI*wout with RPNN
%
function Fmin = funPIloss(w,Aa,bb,Xin,fixedPar)
    [inSz,np] = size(Xin);

    %% get random projection matrix
    PHI = getRPmatrix(Aa, bb, Xin);
    
    %% calculate RPNN output: Wout*PHI
    hfunRPNN = w*PHI;

    %% get the RHS of MModeSS: f and g
    kappa = fixedPar(1);
    sigma = fixedPar(2);
    dydtMM = MModeSS(0., [hfunRPNN(1,:); Xin(1:inSz-1,:)], Xin(inSz,:), kappa, sigma); % autonomous system
    f = dydtMM(1:inSz-1,:).*Xin(inSz,:);        % f = dxdt * eps
    g = dydtMM(inSz,:);                         % g = dydt
    
    %% get grad_y(h(y,e)) = grad_y(PHI*wout)
    [dNNdy, ~] = gradNN_y1(PHI,Aa,w);
    
    %% compute loss we want to minimize as (Cx1) functions
    Fmin = f-Xin(inSz,:).*dNNdy.*g;          % f - eps * grad_y (h(y,e) * g
    
end

%% Derivatives of the Physics-Informed residuals with respect to the output weights of the RPNN
% form  grad f_w - e* grad (grad x_y)_w * g - e* grad x_y * grad g_w
% x = h(y,e) = PHI*wout with RPNN
%
function dFdW = gradFminWout(w,Aa,bb,Xin,fixedPar)
    [inSz,np] = size(Xin);
    
    %% get random projection matrix
    PHI = getRPmatrix(Aa, bb, Xin);

    %% calculate RPNN output: Wout*PHI
    hfunRPNN = w*PHI;
    
    %% get grad f_w and grad g_w 
    [gradFastW, gradSlowW] = gradRHSW(w,PHI,Xin,fixedPar);
   
    %% grad_w (grad_y (h(y,e))) which is intermediate for the first (byproduct, due to RPNN)!!!
    [dNNdy, derLog] = gradNN_y1(PHI,Aa,w);

    %% get only Slow RHS  
    g = zeros(1,np);
    kappa = fixedPar(1);
    sigma = fixedPar(2);
    for i = 1:np
        dydtMM = MModeSS(0., [hfunRPNN(1,i); Xin(1,i)], Xin(2,i), kappa, sigma); % autonomous system
        g(1,i) = dydtMM(2);                % g = dydt
    end

    %% form dFdW
    dFdW = gradFastW - Xin(2,:).*(derLog.*g+dNNdy.*gradSlowW); 

end

%% function to calculate grad_y(h(y,e)) with repsect to slow variables ACCORDING TO LOGISTIC SIGMOID activation
% where h = PHI*Wout 
%
% derivative of logistic sigmoid is psi(x)' = psi(x)*(1-psi(x)) 
function [dNNdy, derLog] = gradNN_y1(PHI,Aa,Wout)
    % slow variables confirmation
    [outSz,N] = size(Wout);
    % derivative of activation function vector
    argum1 = PHI.*(1-PHI); % LxC
    derLog = argum1.*Aa(1:N,1); % (LxC)
    
    % dderivative of RPNN 
    dNNdy = Wout*derLog;   % MxC (M=1 in this case)
end

%% function to calculate gradients of the RHS with respect to the output Weights again ACCORDING TO LOGISTIC SIGMOID activation
% gradFastW = grad f_w = grad f_x * grad x_w      f = e*dx/dt
%
function [gradFastW, gradSlowW] = gradRHSW(w,PHI,Xin,fixedPar)
    [inSz,np] = size(Xin);

    %% calculate RPNN output: Wout*PHI
    hfunRPNN = w*PHI;
    
    %% get analytic derivatives of f and g over x 
    gradfx = zeros(1,np);
    gradgx = zeros(1,np);
    kappa = fixedPar(1);
    sigma = fixedPar(2);
    [Jac_x, ~] = gradMModeSS(0.,[hfunRPNN(1,:); Xin(1:inSz-1,:)], Xin(inSz,:), kappa, sigma);
    gradfx(1,:) = Jac_x(1,:).*Xin(2,:);
    gradgx(1,:) = Jac_x(2,:);

    %%
    gradFastW = gradfx.*PHI;
    gradSlowW = gradgx.*PHI;
end


%% function to get PHI from Aa, bb and Yin: USING LOGISTIC SIGMOID activation
function PHI = getRPmatrix(Aa, bb, Xin)
    % Retrieve Random Projection Matrix PHI (CxL)
    PHI = logsigmoidP(Xin,Aa,bb);
end


%% activation function: parameteric logistic sigmoid for each xi (dx1)
%
%  ai: is 1xd  and bi = 1x1 
function psi = logsigmoidP(xi,ai,bi)
    Z=ai*xi+bi;
    psi = 1./(1+exp(-Z));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
