clear
clc

rng default; % for reproducibility

%% set dimensions before calling the generating function
inSz = 2;           % number of slow variables + 1
outSz = 1;          % number of fast variables
fDS = 2;            % 1: with FD, 2: with SD, 3: with AD
noiterations = 1;   % number of training runs to consider

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
learnPar = zeros(noiterations,81);
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
    %% construct NN and randomly initialize it
    noHL = 1;       % one hidden layer
    N = 20;      % number of neurons per hidden layer
    % input layer output: Win*Y+bin (per sample)
    Win = 2*rand(N,inSz)-1;      % input weights          
    bin = 2*rand(N,1)-1;
    Wout = 2*rand(1,N)-1;
    bout = 2*rand()-1;

    %% include learnables in one array (for transfering)
    learnables = [reshape(Win,1,[]) reshape(bin,1,[]) reshape(Wout,1,[]) reshape(bout,1,[])];
    %% and dimensions in another array
    netDim = [inSz noHL(1) N outSz]; % if you have more hidden layers include them as arrays
    
    %% solve the system f(x) = 0, where x are the learnables
    %
    % construct the loss function to be minimized
    tic
    fixedPar = [kappa sigma];
    %% solve with FD
    if fDS == 1  
        options=optimoptions(@lsqnonlin,'Display','none','MaxIter',500,'TolFun',1e-03,...
            'Algorithm','levenberg-marquardt','MaxFunEvals',200000,'FinDiffRelStep',1E-07,...
            'SpecifyObjectiveGradient',false,'ScaleProblem','none','StepTolerance',1e-7,'UseParallel',false);
        [learned,resnorm,residual,exitflag,output,lambda,jacobian] = ...
        lsqnonlin(@(curLearn) funPIloss(curLearn,netDim,Xtrain,fixedPar),learnables,[],[],options);
    %% solve with SD
    elseif fDS == 2
        options=optimoptions(@lsqnonlin,'Display','none','MaxIter',30,'TolFun',1e-05,...
            'Algorithm','levenberg-marquardt','MaxFunEvals',200000,...     'CheckGradients',true,...
            'SpecifyObjectiveGradient',true,'ScaleProblem','none','StepTolerance',1e-9,'UseParallel',false);
        [learned1,resnorm,residual,exitflag,output,lambda,jacobian] = ...
        lsqnonlin(@(curLearn) funPIloss(curLearn,netDim,Xtrain,fixedPar),learnables,[],[],options);
        options.ScaleProblem = 'jacobian';
        options.MaxIterations = 500;
        options.FunctionTolerance = 1e-3;
        options.StepTolerance = 1e-7;
        [learned,resnorm,residual,exitflag,output,lambda,jacobian] = ...
        lsqnonlin(@(curLearn) funPIloss(curLearn,netDim,Xtrain,fixedPar),learned1,[],[],options);

    %% solve with AD
    elseif fDS ==3
        LPs = optimvar('LPs',size(learnables,2));
        [Win, bin, Wout, bout] = unravelLearn(LPs,netDim);
         
        Yin=Xtrain;
        phiL = activationFun(Win*Yin+bin*ones(1,size(Yin,2)));
        NNout = Wout*phiL+bout*ones(1,size(Yin,2));
        fMM = (Yin(1,:)-NNout(1,:).*(kappa+Yin(1,:))/(kappa+1));
        gMM = (1/sigma)*(-(kappa+1)*Yin(1,:)+(kappa-sigma+Yin(1,:)).*NNout(1,:));
        dNNdyout = Wout*(((Win(:,1)*ones(1,size(Yin,2)))).*(phiL.*(1-phiL)));

        fun = fMM-Yin(2,:).*dNNdyout.*gMM;
        %% form problem
        eq1 = fun == zeros(outSz,size(Yin,2));
        prob = eqnproblem;
        prob.Equations.eq1 = eq1;

        LPs0.LPs = learnables;
        options=optimoptions(@lsqnonlin,'Display','none','MaxIterations',500,'FunctionTolerance',1e-03,...
            'Algorithm','levenberg-marquardt','MaxFunctionEvaluations',1000000,'StepTolerance',1e-7);
        [sol,fval,exitflag] = solve(prob,LPs0,'Options',options,'solver','lsqnonlin');  
        learned = sol.LPs';
        residual = fval.eq1;
    end
    CPUend = toc;
    CPUrecs(i,1) = CPUend;
    learnPar(i,:) = learned;

    %% PINN train error and CV errors
    trainMSE(i,1) = mse(residual,zeros(size(Ytrain)));
    trainLinf(i,1) = norm(residual,Inf);
    trainL2(i,1) = norm(residual,2);
    CVres = funPIloss(learned,netDim,XCV,fixedPar);
    CVMSE(i,1) = mse(CVres,zeros(size(YCV)));
    CVLinf(i,1) = norm(CVres,Inf);
    CVL2(i,1) = norm(CVres,2);

    %% Evaluation on test data (which are on SIM)
    testSIM = forwardNN(Xtest,learned,netDim);
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
save learned_PI_SLFNN bestLearned;

%% Metrics on training
fprintf('-------PIML SLFNN  TRAINING metrics --------\n')
fprintf('CPU times:    mean              min              max  \n');
fprintf('         %e      %e     %e     \n',mean(CPUrecs),min(CPUrecs),max(CPUrecs));
fprintf('Train err:     l2              lInf              MSE  \n');
fprintf('         %e      %e     %e     \n',mean(trainL2),mean(trainLinf),mean(trainMSE));
fprintf('CV    err:     l2              lInf              MSE  \n');
fprintf('         %e      %e     %e     \n',mean(CVL2),mean(CVLinf),mean(CVMSE));

%% Metrics on testing (on SIM data)
fprintf('-------PIML SLFNN  TESTING metrics --------\n')
fprintf('Test  err:     l2              lInf              MSE  \n');
fprintf('         %e      %e     %e     \n',mean(testL2),mean(testLinf),mean(testMSE));

return


%%% END OF MAIN ROUTINE


% Physics-Informed Loss Function to be minimized for learning NN
%
% solve f(h(y,e),y,e) = e*grad_y(h(y,e))*g(h(y,e),y,e)    where f = e*dx/dt and g = dy/dt 
%
function [Fmin, Jac] = funPIloss(curLearn,netDim,Yin,fixedPar)
    % calculate NN output
    hfunNN = forwardNN(Yin,curLearn,netDim);
    %
    % get from MModeSS f and g
    inSz = netDim(1);
    outSz = netDim(4);
    nSample = size(Yin,2);
    dydtMM = zeros(inSz,nSample);
    f = zeros(outSz,nSample);
    g = zeros(inSz-1,nSample);
    kappa = fixedPar(1);
    sigma = fixedPar(2);
    for i = 1:nSample
        dydtMM(:,i) = MModeSS(0., [hfunNN(i); Yin(1,i)], Yin(2,i), kappa, sigma); % autonomous system
        f(1,i) = dydtMM(1,i)*Yin(2,i);       % f = dxdt * eps
        g(1,i) = dydtMM(2,i);                % g = dydt
    end
    %
    % get grad_y(h(y,e))
    dNNdy = gradNN_y1(Yin,curLearn,netDim);
    %
    % compute loss
    Fmin = f-Yin(2,:).*dNNdy.*g;          % f - eps * grad_y (h(y,e) * g
    
    if nargout>1
        % get gradients from RHS 
        gradfx = zeros(outSz,nSample);
        gradgx = zeros(inSz-1,nSample);
        for i = 1:nSample
            [Jac_x, ~] = gradMModeSS(0.,[hfunNN(i); Yin(1,i)],Yin(2,i), kappa, sigma);
            gradfx(1,i) = Jac_x(1,1)*Yin(2,i);
            gradgx(1,i) = Jac_x(2,1);
        end
        % get gradients of NN w.r.t. parameters
        [dNNdWo, ddNNdWody, dNNdBo, ddNNdBody, dNNdW, ddNNdWdy, dNNdB, ddNNdBdy] = gradsNN(Yin, curLearn, netDim); 
        %% all together
        temp1 = [dNNdW; dNNdB; dNNdWo; dNNdBo];
        temp2 = [ddNNdWdy; ddNNdBdy; ddNNdWody; ddNNdBody];
        Jac = (gradfx.*temp1 - Yin(2,:).*(temp2.*g+dNNdy.*gradgx.*temp1))';
        
    end
end

% function to calculate the gradients of the NN w.r.t. parameters
function [dNNdWo, ddNNdWody, dNNdBo, ddNNdBody, dNNdW, ddNNdWdy, dNNdB, ddNNdBdy] = gradsNN(Yin, learnables, netDim)
    [Win, bin, Wout, bout] = unravelLearn(learnables,netDim);
    %
    phiL = activationFun(Win*Yin+bin);
    dphiL = phiL.*(1-phiL);
    dNNdWo = phiL;                     % per i,j dim: M x ML
    ddNNdWody = Win(:,1).*dphiL;       % per i,j dim: M x ML x d for d=1...N-M
    %
    dNNdB = Wout'.*dphiL;                                         % per i,j dim: M x ML
    %ddNNdBdy = Wout'.*Win(:,1).*phiL.*(1-phiL).*(1-2*phiL);
    ddNNdBdy = dNNdB.*Win(:,1).*(1-2*phiL);                       % per i,j dim: M x ML x d for d=1...N-M
    %
    dNNdBo = ones(1,size(Yin,2));
    ddNNdBody = zeros(1,size(Yin,2));
    %
    %dNNdW1 = Wout'.*dphiL.*Yin(1,:);                                % per i,j dim: M x M(N-M)L
    dNNdW1 = dNNdB.*Yin(1,:); 
    %ddNNdWdy1 = Wout'.*dphiL.*(Win(:,1).*(1-2*phiL).*Yin(1,:)+1);   % per i,j dim: M x M(N-M)L x d
    ddNNdWdy1 = dNNdB.*(Win(:,1).*(1-2*phiL).*Yin(1,:)+1);
    %dNNdW2 = Wout'.*dphiL.*Yin(2,:);                                % per i,j dim: M x M 1 L           for epsilon
    dNNdW2 = dNNdB.*Yin(2,:); 
    %ddNNdWdy2 = Wout'.*dphiL.*(Win(:,1).*(1-2*phiL).*Yin(2,:));     % per i,j dim: M x M 1 L x d       for epsilon
    ddNNdWdy2 = ddNNdBdy.*Yin(2,:);

    dNNdW = [dNNdW1; dNNdW2];  % concatenated
    ddNNdWdy = [ddNNdWdy1; ddNNdWdy2];  % concatenated
    
end

% function to calculate grad_y(h(y,e))
function dNNdy = gradNN_y1(Yin,learnables,netDim)
    [Win, bin, Wout, ~] = unravelLearn(learnables,netDim);
    %
    phiL = activationFun(Win*Yin+bin);
    % derivative of activation function
    argum2 = phiL.*(1-phiL);           % for logsig
    dNNdy = Wout*(Win(:,1).*argum2);   % only the Win refering to y and not eps
end

% function for calculating NN output
function NNout = forwardNN(Yin,learnables,netDim)
    [Win, bin, Wout, bout] = unravelLearn(learnables,netDim);

    %% forward the input
    argum1 = Win*Yin+bin;
    NNout = Wout*activationFun(argum1)+bout;

end

% function to unravel learnable parameters
function [Win, bin, Wout, bout] = unravelLearn(learnables,netDim)
    inSz = netDim(1);
    HLsz = netDim(end-1);
    outSz = netDim(end);
    noHL = netDim(2:end-2);
    
    if isfloat(learnables)
        %% unravel learnables
        dummy = learnables;
        Win = reshape(dummy(1:inSz*HLsz),[HLsz, inSz]);
        dummy(1:inSz*HLsz) = [];
        bin = reshape(dummy(1:HLsz),[HLsz, 1]);
        dummy(1:HLsz) = [];
        if noHL>1
            % here include additional hidden layers
        end
        Wout = reshape(dummy(1:HLsz*outSz),[outSz, HLsz]);
        dummy(1:HLsz*outSz) = [];
        bout = reshape(dummy(1:outSz),[outSz, 1]);
        dummy(1:outSz) = [];
        if ~isempty(dummy); error('Wrong unraveling in unravelLearn function'); end
    %% to handle optimpars
    else 
        Win = reshape(learnables(1:inSz*HLsz),[HLsz, inSz]);
        bin = reshape(learnables(inSz*HLsz+1:inSz*HLsz+HLsz),[HLsz, 1]);
        Wout = reshape(learnables(inSz*HLsz+HLsz+1:inSz*HLsz+HLsz+HLsz*outSz),[outSz, HLsz]);
        bout = reshape(learnables(inSz*HLsz+HLsz+HLsz*outSz+1:inSz*HLsz+HLsz+HLsz*outSz+outSz),[outSz, 1]);
        if inSz*HLsz+HLsz+HLsz*outSz+outSz~=size(learnables,1); error('Wrong unraveling in unravelLearn function'); end
    end

end

%% activation function
%
% seperate so you can change it wherever
function s=activationFun(x)
   s = logsig(x);
end
