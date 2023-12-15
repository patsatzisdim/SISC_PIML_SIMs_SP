clear
clc

rng default; % for reproducibility

%% set dimensions before calling the generating function
inSz = 3;           % number of slow variables + 1
outSz = 1;          % number of fast variables
fDS = 3;            % 1: with FD, 2: with SD, 3: with AD
noiterations = 1;   % number of training runs to consider

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

%% load training, test data
load TMDDTest allData;
Xtest = allData(end-inSz+1:end,:);
Ytest = allData(1,:);

load TMDDTrain allData;
Xtrain = allData(end-inSz+1:end,:);
Ytrain = allData(1,:);

%% Testing Set: accuracy of GSPT analytic expressions
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
%
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
learnPar = zeros(noiterations,101);
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

    %% include learnables in one array (for transfer)
    learnables = [reshape(Win,1,[]) reshape(bin,1,[]) reshape(Wout,1,[]) reshape(bout,1,[])];
    %% and dimensions in another array
    netDim = [inSz noHL(1) N outSz]; % if you have more hidden layers include them as arrays
    
    %% solve the system f(x) = 0, where x are the learnables
    %
    % construct the loss function to be minimized
    tic
    fixedPar = [k1 k2 k3 k4];
    %% solve with FD
    if fDS == 1  
        options=optimoptions(@lsqnonlin,'Display','none','MaxIter',500,'TolFun',1e-03,...
            'Algorithm','levenberg-marquardt','MaxFunEvals',200000,'FinDiffRelStep',1E-09,...
            'SpecifyObjectiveGradient',false,'ScaleProblem','none','StepTolerance',1e-9,'UseParallel',false);
        [learned,resnorm,residual,exitflag,output,lambda,jacobian] = ...
        lsqnonlin(@(curLearn) funPIloss(curLearn,netDim,Xtrain,fixedPar),learnables,[],[],options);
    %% solve with SD
    elseif fDS == 2
        options=optimoptions(@lsqnonlin,'Display','none','MaxIter',50,'TolFun',1e-05,...
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
        phiL = actFun(Win*Yin+bin*ones(1,size(Yin,2)));
        NNout = Wout*phiL+bout*ones(1,size(Yin,2));
        fTMDD = (-NNout(1,:).*Yin(1,:)+k1*Yin(2,:)+1) - Yin(3,:)*k2.*NNout(1,:); 
        gTMDD = [k3*(-NNout(1,:).*Yin(1,:)+k1*Yin(2,:))-k4*Yin(1,:);
                k2*(NNout(1,:).*Yin(1,:)-k1*Yin(2,:))-Yin(2,:)];      
        dNNdyout = [Wout*(((Win(:,1)*ones(1,size(Yin,2)))).*(phiL.*(1-phiL)));
                    Wout*(((Win(:,2)*ones(1,size(Yin,2)))).*(phiL.*(1-phiL)))];

        fun = fTMDD-Yin(3,:).*(dNNdyout(1,:).*gTMDD(1,:)+dNNdyout(2,:).*gTMDD(2,:));
        %% form problem
        eq1 = fun == zeros(outSz,size(Yin,2));
        prob = eqnproblem;
        prob.Equations.eq1 = eq1;

        LPs0.LPs = learnables;
        options=optimoptions(@lsqnonlin,'Display','none','MaxIterations',500,'FunctionTolerance',1e-03,...
            'Algorithm','levenberg-marquardt','MaxFunctionEvaluations',1000000,'StepTolerance',1e-7);
        [sol,fval,exitflag] = solve(prob,LPs0,'Options',options,'solver','lsqnonlin');   % AD by default!!!!!
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
function [Fmin, Jac] = funPIloss(curLearn,netDim,X,fixedPar)
    % calculate NN output
    hfunNN = forwardNN(X,curLearn,netDim);
    %
    % get from TMDDodeSPA f and g
    inSz = netDim(1);
    outSz = netDim(4);
    ns = size(X,2);
    f = zeros(outSz,ns);
    g = zeros(inSz-1,ns);
    k1 = fixedPar(1);
    k2 = fixedPar(2);
    k3 = fixedPar(3);
    k4 = fixedPar(4);

    dydtTMDD = TMDDodeSPA(0.,[hfunNN; X(1:2,:)],X(3,:),k1,k2,k3,k4); % autonomous system
    f(1,:) = dydtTMDD(1,:).*X(3,:);       % f = dxdt * eps
    g(1:2,:) = dydtTMDD(2:3,:);                % g = dydt
    %
    % get grad_y(h(y,e))
    [dNNdy, ~] = gradNN_y1(X,curLearn,netDim);
    %
    % compute loss
    Fmin = f-X(3,:).*sum(dNNdy.*g);          % f - eps * grad_y (h(y,e) * g

    if nargout>1
        % get gradients from RHS 
        [Jac_x, ~] = gradTMDDodeSPA(0.,[hfunNN; X(1:2,:)],X(3,:),k1,k2,k3,k4);
        gradfx = Jac_x(1,:).*X(3,:);
        gradgx = Jac_x(2:3,:);
        % get gradients of NN w.r.t. parameters
        [dNNdWo, ddNNdWody, dNNdBo, ddNNdBody, dNNdW, ddNNdWdy, dNNdB, ddNNdBdy] = gradsNN(X, curLearn, netDim); 
        %% all together
        temp1 = [dNNdW; dNNdB; dNNdWo; dNNdBo];
        temp2 = [ddNNdWdy; ddNNdBdy; ddNNdWody; ddNNdBody];
        Jac = (gradfx.*temp1 - X(3,:).*(squeeze(temp2(:,1,:)).*g(1,:)+dNNdy(1,:).*gradgx(1,:).*temp1+ ...
                                        squeeze(temp2(:,2,:)).*g(2,:)+dNNdy(2,:).*gradgx(2,:).*temp1))';
    end
    
end

% function to calculate the gradients of the NN w.r.t. parameters
function [dNNdWo, ddNNdWody, dNNdBo, ddNNdBody, dNNdW, ddNNdWdy, dNNdB, ddNNdBdy] = gradsNN(Yin, learnables, netDim)
    [Win, bin, Wout, bout] = unravelLearn(learnables,netDim);
    hlSz = size(Win,1);
    inSz = size(Win,2);
    Ns = size(Yin,2);
    %
    phiL = actFun(Win*Yin+bin);
    dphiL = phiL.*(1-phiL);
    dNNdWo = phiL;                       % per i,j dim: M x ML
    ddNNdWody = zeros(hlSz,inSz-1,Ns);   % per i,j dim: M x ML x d for d=1...N-M
    for i=1:inSz-1
        ddNNdWody(:,i,:) = Win(:,i).*dphiL;    
    end
    %
    dNNdB = Wout'.*dphiL;                                         % per i,j dim: M x ML
    ddNNdBdy = zeros(hlSz,inSz-1,Ns);                             % per i,j dim: M x ML x d for d=1...N-M
    for i=1:inSz-1
        ddNNdBdy(:,i,:) = dNNdB.*Win(:,i).*(1-2*phiL);   
    end
    %ddNNdBdy = Wout'.*Win(:,1).*phiL.*(1-phiL).*(1-2*phiL);
    %
    dNNdBo = ones(1,size(Yin,2));
    ddNNdBody = zeros(1,inSz-1,size(Yin,2));
    %
    dNNdW = zeros(inSz*hlSz,Ns);                                     % per i,j dim: M x MDL, that is both N-M and epsilon
    ddNNdWdy = zeros(inSz*hlSz,inSz-1,Ns);                           % per i,j dim: M x MDL x d for d=1...N-M
    for j=1:inSz % j=inSz corresponds to epsilon        
        %dNNdW1 = Wout'.*dphiL.*Yin(i,:);                            
        dNNdW(hlSz*(j-1)+1:hlSz*j,:) = dNNdB.*Yin(j,:); 
        for i=1:inSz-1   % i samples d=1,N-M
            if i==j  % when d=h
                %ddNNdWdy1 = Wout'.*dphiL.*(Win(:,i).*(1-2*phiL).*Yin(j,:)+1);   % per i,j dim: M x M D L x d
                ddNNdWdy(hlSz*(j-1)+1:hlSz*j,i,:) = dNNdB.*(Win(:,i).*(1-2*phiL).*Yin(j,:)+1);
            else    % when d~=h or for all epsilons
                ddNNdWdy(hlSz*(j-1)+1:hlSz*j,i,:) = dNNdB.*(Win(:,i).*(1-2*phiL).*Yin(j,:));
            end
        end
    end   
end

% function to calculate grad_y(h(y,e))
function [dNNdy, derLog] = gradNN_y1(X,learnables,netDim)
    [Win, bin, Wout, ~] = unravelLearn(learnables,netDim);
    %
    inSz = netDim(1);
    N = netDim(3);
    argum1 = actFun(Win*X+bin);
    % derivative of activation function
    argum2 = argum1.*(1-argum1); % for logsig
    for i=1:inSz-1
        derLog(:,:,i) = argum2.*Win(1:N,i); % (LxC)
        % dderivative of PISLFNN 
        dNNdy(i,:)= Wout*derLog(:,:,i);
    end
end

% function for calculating NN output
function NNout = forwardNN(Yin,learnables,netDim)
    [Win, bin, Wout, bout] = unravelLearn(learnables,netDim);

    %% forward the input
    argum1 = Win*Yin+bin;
    NNout = Wout*actFun(argum1)+bout;

end

% function to unravel learnable parameters
function [Win, bin, Wout, bout] = unravelLearn(learnables,netDim)
    inSz = netDim(1);
    HLsz = netDim(end-1);
    outSz = netDim(end);
    noHL = netDim(2:end-2);

    %% unravel learnables
    if isfloat(learnables)
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
function s=actFun(x)
   s = logsig(x);
end

