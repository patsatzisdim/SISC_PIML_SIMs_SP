clear
clc

set(0,'DefaultLineLineWidth',2);

rng default % for reproducibility

%% set dimensions before calling the generating function
inSz = 3;           % number of slow variables + 1
outSz = 1;          % number of fast variables
noiterations = 1;   % number of training runs to consider

%% Set Random Projections of RPNN
N = 101;      % number of neurons per hidden layer
fRPscale = 1;   

%% set the parameters for the full model
a = 0.1;
b = 0.6;
k = 1;

%% load training, test data
load TLCTest allData;
Xtest = allData(end-inSz+1:end,:);
Ytest = allData(1,:);

load TLCTrain allData;
Xtrain = allData(end-inSz+1:end,:);
Ytrain = allData(1,:);

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
learnPar = zeros(noiterations,N*inSz+N+N);
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
    %% fix internal weights and randomly initialize output layer
    tic
    Wout = (2*rand(outSz,N)-1);    % output weights    1xL
    [Alpha, beta] = RPinternalWeights(N,Xtrain,fRPscale);
    PHI= getRPmatrix(Alpha, beta, Xtrain);

    %% Train RPNN
    fixedPar = [a b k];
    learnables = Wout;
    %% Perform Newton with SVD
    maxIter = 30;           % max iterations allowed
    iter = 0;
    Fmin = funPIloss(learnables,Alpha,beta,Xtrain,fixedPar);    % residuals at the first iteration
    NFminP=norm(Fmin);
    bestnorm=NFminP;
    Cor = 1000;             % for oscillating errors to be avoided
    contfail=0; maxcntf=3;  % allow three times
    while (iter <= maxIter) && (Cor-1>1e-3) && contfail<maxcntf
        iter=iter+1;
        % calculate the residual derivatives w.r.t. out weights
        dFdW = gradFminWout(learnables,Alpha,beta,Xtrain,fixedPar); %% in LxC format
        % solve the system dFdW * dW = - Fmin   with SVD
        %
        dW = -Fmin*pinv(dFdW,1e-8);
        eta=dW;
        learnables = learnables + eta; 
        % calculate new residuals
        Fmin = funPIloss(learnables,Alpha,beta,Xtrain,fixedPar);
        newFminP=norm(Fmin);
        %  check new error
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
    if contfail==maxcntf; fprintf('RPNN may not converged %f. \n',bestnorm); %i=i-1;
    else
        fprintf('RPNN CONVERGE %f. \n',NFminP);
    end
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
%% best w.r.t CV
[aisdjoasi,idx_bnet]=min(CVL2);
btestMSE = testMSE(idx_bnet,1);
btestLinf = testLinf(idx_bnet,1);
btestL2 = testL2(idx_bnet,1);
save learned_PI_RPNN bestLearned;

%% Metrics on training
fprintf('-------PIML PIRPNN  TRAINING metrics --------\n')
fprintf('CPU times:    mean              min              max  \n');
fprintf('         %e      %e     %e     \n',mean(CPUrecs),min(CPUrecs),max(CPUrecs));
fprintf('Train err:     l2              lInf              MSE  \n');
fprintf('         %e      %e     %e     \n',mean(trainL2),mean(trainLinf),mean(trainMSE));
fprintf('CV    err:     l2              lInf              MSE  \n');
fprintf('         %e      %e     %e     \n',mean(CVL2),mean(CVLinf),mean(CVMSE));

%% Metrics on testing (on SIM data)
fprintf('-------PIML PIRPNN  TESTING metrics --------\n')
fprintf('Test  err:     l2              lInf              MSE  \n');
fprintf('         %e      %e     %e     \n',mean(testL2),mean(testLinf),mean(testMSE));
%% bestCV Metrics on testing (on SIM data)
fprintf('-------PIML PIRPNN best CV TESTING metrics --------\n')
fprintf('Test  err:     l2              lInf              MSE  \n');
fprintf('         %e      %e     %e     \n',btestL2,btestLinf,btestMSE);

return


%%% END OF MAIN ROUTINE


%% FUNCTIONS


%% function to randomly determine the internal weights ACCORDING TO LOGISTIC SIGMOID activation 
% inSz: input size, hlSz: hidden layer size, ranges: minmax of each input
% (miny1 maxy1 mineps maxeps here)
function [Aa, bb, Centr] = RPinternalWeights(N,X,RPScale)
    Xmin=min(X,[],2);
    Xmax=max(X,[],2);
    DX=Xmax-Xmin;
    [d,np]=size(X);
    % fix centroids uniformly along ranges (COULD BE LOGSPACED)
    if RPScale == 1 % uniform scaling
        Centr=[DX(1:d-1)'.*rand(N,d-1)+Xmin(1:d-1)',...
            logspace(log10(Xmin(d))-1,log10(Xmax(d))+1,N)']; 
        Aa = 2*(rand(N,d)-0.5);
    end
    %% Set bb so that inflection point being at the center
    bb= -sum(Aa.*Centr,2);
end

%% Physics-Informed Loss Function to be minimized for learning RPNN
% form f(h(y,e),y,e) - e*grad_y(h(y,e))*g(h(y,e),y,e)    where f = e*dx/dt and g = dy/dt 
%
% h(y,e) = PHI*wout with RPNN
function Fmin = funPIloss(w,Aa,bb,Xin,fixedPar)
    %[outSz,N] = size(w);
    [inSz,~] = size(Xin);

    %% get random projection matrix
    PHI = getRPmatrix(Aa, bb, Xin);
    
    %% calculate RPNN output: Wout*PHI
    hfunRPNN = w*PHI;

    %% get the RHS of MModeSS: f and g
    a = fixedPar(1);
    b = fixedPar(2);
    k = fixedPar(3);
    dydtToyLC = ToyLCode_SP2(0.,[hfunRPNN(1,:); Xin(1:2,:)],Xin(3,:),a,b,k); % autonomous system
    f(1,:) = dydtToyLC(1,:).*Xin(3,:);       % f = dxdt * eps
    g(1:2,:) = dydtToyLC(2:3,:);                % g = dydt
    
    %% get grad_y(h(y,e)) = grad_y(PHI*wout)
    [dNNdy, ~] = gradNN_y1(PHI,Aa,w);
    
    %% compute loss we want to minimize as (Cx1) functions
    Fmin = f-Xin(3,:).*sum(dNNdy.*g);          % f - eps * grad_y (h(y,e) * g
    
end

%% Derivatives of the Physics-Informed residuals with respect to the output weights of the RPNN
% form  grad f_w - e* grad (grad x_y)_w * g - e* grad x_y * grad g_w
% x = h(y,e) = PHI*wout with RPNN
%
function dFdW = gradFminWout(w,Aa,bb,Xin,fixedPar)
    %[outSz,N] = size(w);
    [inSz,np] = size(Xin);
    
    %% get random projection matrix
    PHI = getRPmatrix(Aa, bb, Xin);

    %% calculate RPNN output: Wout*PHI
    hfunRPNN = w*PHI;
    
   
    %% grad_w (grad_y (h(y,e))) which is intermediate for the first (byproduct, due to RPNN)!!!
    [dNNdy, derLog] = gradNN_y1(PHI,Aa,w);

    %% get only Slow RHS
    a = fixedPar(1);
    b = fixedPar(2);
    k = fixedPar(3);
   
    dydtToyLC =  ToyLCode_SP2(0.,[hfunRPNN(1,:); Xin(1:2,:)],Xin(3,:),a,b,k); % autonomous system
    g(1:2,:) = dydtToyLC(2:3,:);                % g = dydt

    %% get derivatives df/dx, dg/dx
    [Jac_x, ~] = gradToyLCode_SP(0.,[hfunRPNN(1,:); Xin(1:2,:)],Xin(3,:),a,b,k);
    gradfx = Jac_x(1,:).*Xin(3,:);
    gradgx = Jac_x(2:3,:);
    dNNdWo = PHI;
    ddNNdWody = derLog;

    %% form Jacobian w.r.t parameters
    dFdW = gradfx.*dNNdWo - Xin(3,:).*(squeeze(ddNNdWody(:,:,1)).*g(1,:)+dNNdy(1,:).*gradgx(1,:).*dNNdWo+ ...
                                    squeeze(ddNNdWody(:,:,2)).*g(2,:)+dNNdy(2,:).*gradgx(2,:).*dNNdWo);

end

%% function to calculate grad_y(h(y,e)) with repsect to slow variables ACCORDING TO LOGISTIC SIGMOID activation
% where h = PHI*Wout 
%
% derivative of logistic sigmoid is psi(x)' = psi(x)*(1-psi(x)) 
function [dNNdy, derLog] = gradNN_y1(PHI,Aa,Wout)
    % slow variables confirmation
    [~,N] = size(Wout);
    inSz=size(Aa,2);
    np=size(PHI,2);
    % derivative of activation function vector
    argum1 = PHI.*(1-PHI); % LxC
    derLog=zeros(N,np,inSz-1);
    dNNdy=zeros(inSz-1,np);
    for i=1:inSz-1
        derLog(:,:,i) = argum1.*Aa(1:N,i); % (LxC)
        % dderivative of RPNN 
        dNNdy(i,:)= Wout*derLog(:,:,i);
    end
end

%% function to get PHI from Aa, bb and Yin: USING LOGISTIC SIGMOID activation
function PHI = getRPmatrix(Aa, bb, Xin)
    PHI = logsigmoidP(Xin,Aa,bb);
end


%% activation function: parameteric logistic sigmoid for each xi (dx1)
%
%  ai: is 1xd  and bi = 1x1 
function psi = logsigmoidP(xi,ai,bi)
    Z=ai*xi+bi;
    psi = 1./(1+exp(-Z));
end