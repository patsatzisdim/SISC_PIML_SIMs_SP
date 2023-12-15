clear
clc

%% PLOTTING SCRIPT
fPlot = 1;      % 1: for the SIM in phase space
                % 2-5: error plots of QSSA, GSPT Oe1, GSPT Oe2, CSP
                % 6-7: error plots of SLFNN, RPNN

if fPlot == 1
    %% load testing data that are on SIM
    inSz = 2;               % number of slow variables + 1
    outSz = 1;              % number of fast variables
    nSamples = 400;
    nICs = 10;
    load MMTestEps1e-2 allData;
    Ydata = allData(2,:);
    Epsdata = allData(3,:);
    Xdata = allData(1,:);

    %% sort to visulize as line
    [~, idx] = sort(Xdata,'descend');
    Xdata = Xdata(1,idx);
    Ydata = Ydata(1,idx);
    
    %% show some trajectories approaching the SIM
    KM = 1e-2;
    K = 1e-1;
    s0 = 1e-3;
    c0 = 0.;
    kappa = KM/s0;
    sigma = K/s0;
    %
    eps2 = 1e-2;
    parVec = [kappa sigma eps2];
    y_init1 = [1e-3 ; 1e-1];
    [yAll1, tSol1] = getMMdata(parVec,0,y_init1,false,0);
    y_init2 = [1e-2 ; 1e-3];
    [yAll2, tSol2] = getMMdata(parVec,0,y_init2,false,0);
    

    %%
    figure(1);
    ax = axes('OuterPosition',[0 0 1 1]);
    plot(ax,Ydata,Xdata,'b-','LineWidth',5); hold on;
    scatter(ax,yAll1(2,1),yAll1(1,1),100,'rs','filled'); hold on;
    plot(ax,yAll1(2,:),yAll1(1,:),'r-','LineWidth',2); hold on;
    scatter(ax,yAll2(2,1),yAll2(1,1),100,'rs','filled'); hold on;
    plot(ax,yAll2(2,:),yAll2(1,:),'r-','LineWidth',2); hold off;

    ax.XScale = 'log';
    ax.YScale = 'log';
    ax.XLim = [1e-6 1];
    ax.YLim = [1e-6 1];
    ax.XTick = logspace(-6,0,4);
    ax.YTick = logspace(-6,0,4);
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';

    ax.FontName = 'times';
    ax.FontSize = 20;
    ax.LabelFontSizeMultiplier = 24/20;
    ax.TickLabelInterpreter = 'latex';
    ax.XLabel.String = '$y$';
    ax.XLabel.Interpreter = 'latex';
    ax.XLabel.FontSize = 24;
    ax.YLabel.String = '$x$';
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 24;
    ax.YLabel.Rotation = 0;


    prePosition = ax.Position;
    ax.XLabel.Position(1) = 1e-3;
    ax.XLabel.Position(2) = 3e-7;   
    ax.YLabel.Position(2) = 0.0005;
    ax.Position  = prePosition;

elseif (fPlot >= 2) && (fPlot<=5)
    %% load testing data
    inSz = 2;   % number of slow variables + 1
    outSz = 1;  % number of fast variables
    nSamples = 100;
    nICs = 5;
    load MMTest allData;
    Ydata = allData(2,:);
    Epsdata = allData(3,:);
    Xdata = allData(1,:);
    
    %% params for SPT/GSPT approximations
    KM = 1e-2;
    K = 1e-1;
    s0 = 1e-3;
    c0 = 0.;
    kappa = KM/s0;
    sigma = K/s0;

    if fPlot == 2
        % oe0
        QSSAo0SIM = Ydata(1,:).*(kappa+1)./(kappa+Ydata(1,:));   % sQSSA
        AErr = abs(Xdata-QSSAo0SIM);
    elseif fPlot == 3
        % oe0
        QSSAo0SIM = Ydata(1,:).*(kappa+1)./(kappa+Ydata(1,:));   % sQSSA
        % oe1
        GSPT_o1C = (kappa*(kappa+1)^3.*Ydata(1,:))./((kappa+Ydata(1,:)).^4);
        o1_GSPT = QSSAo0SIM + Epsdata.*GSPT_o1C;
        AErr = abs(Xdata-o1_GSPT);
    elseif fPlot == 4  
        % oe0
        QSSAo0SIM = Ydata(1,:).*(kappa+1)./(kappa+Ydata(1,:));   % sQSSA
        % oe1
        GSPT_o1C = (kappa*(kappa+1)^3.*Ydata(1,:))./((kappa+Ydata(1,:)).^4);
        % oe2
        GSPT_o2C = -(kappa*(kappa+1)^5.*Ydata(1,:).*(kappa^2+3*sigma*Ydata(1,:)+kappa*(Ydata(1,:)-...
                2*sigma)))./(sigma*(kappa+Ydata(1,:)).^7);
        o2_GSPT = QSSAo0SIM + Epsdata.*GSPT_o1C + Epsdata.^2.*GSPT_o2C;
        AErr = abs(Xdata-o2_GSPT);
    elseif fPlot == 5
        % CSP results
        CSPo2 = (sigma*(kappa + Ydata(1,:)).^2 + Epsdata(1,:).*(1 + kappa)^2.*(kappa - sigma + 2*Ydata(1,:)) - ... 
            sqrt(Epsdata(1,:).^2*(1 + kappa)^4*(kappa - sigma)^2 + sigma^2*(kappa + Ydata(1,:)).^4 + ...
            2*Epsdata(1,:).*(1 + kappa)^2*sigma.*(kappa + Ydata(1,:)).*(kappa*(kappa - sigma) + ...
            (kappa + sigma).*Ydata(1,:))))./(2.*Epsdata(1,:)*(1 + kappa).*(kappa - sigma + Ydata(1,:)));
        AErr = abs(Xdata-CSPo2);
    end

    %%
    figure(1);
    ax = axes('OuterPosition',[0 0 1 1]);
    scatter(ax,Ydata(1,:),Epsdata,50,log10(AErr),'.');
    c = colorbar;
    colormap(jet)
    ax.XScale = 'log';
    ax.YScale = 'log';
    ax.XLim = [8e-7 1]; 
    ax.YLim = [1e-4 1e-1];
    ax.XTick = logspace(-6,0,4);
    ax.YTick = logspace(-4,-1,4);
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    caxis(ax,[-12 -1]);
    c.Ticks = -11:3:-2;
    c.Label.String = 'log(AE)';
    c.Label.Interpreter = 'latex';
    c.Label.FontSize = 20;
    
    ax.FontName = 'times';
    ax.FontSize = 20;
    ax.LabelFontSizeMultiplier = 24/20;
    ax.TickLabelInterpreter = 'latex';
    ax.XLabel.String = '$y$';
    ax.XLabel.Interpreter = 'latex';
    ax.XLabel.FontSize = 24;
    ax.YLabel.String = '$\epsilon$';
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 24;
    ax.YLabel.Rotation = 0;
    
    prePosition = ax.Position;
    ax.XLabel.Position(1) = 1e-3;
    ax.XLabel.Position(2) = 5e-5;
    ax.YLabel.Position(1) = 7e-8;
    ax.YLabel.Position(2) = 2.3e-3;
    ax.Position  = prePosition;

elseif (fPlot>=6) && (fPlot<=7)
    %% load testing data
    inSz = 2;   % number of slow variables + 1
    outSz = 1;  % number of fast variables
    nSamples = 100;
    nICs = 10;
    load MMTest allData;
    Ydata = allData(2,:);
    Epsdata = allData(3,:);
    Xdata = allData(1,:);

    %% forward the testing points through NNs and calculate errors
    if fPlot == 6
        load learned_PI_SLFNN learned;
        netDim = [2 1 20 1];
        preds = forwardNN([Ydata; Epsdata],learned,netDim);
        AErr = abs(Xdata-preds);
    elseif fPlot == 7
        load learned_PI_RPNN learned;
        N = 81;
        d = inSz;
        Wout = learned(1:N);
        Alpha = reshape(learned(N+1:N+N*d),N,[]);
        beta = reshape(learned(N*(d+1)+1:end),N,[]);
        PHI = getRPmatrix(Alpha, beta, [Ydata; Epsdata]);
        preds = Wout*PHI;
        AErr = abs(Xdata-preds);
    end

    %%
    figure(1);
    ax = axes('OuterPosition',[0 0 1 1]);
    scatter(ax,Ydata(1,:),Epsdata,50,log10(AErr),'.');
    c = colorbar;
    colormap(jet)
    ax.XScale = 'log';
    ax.YScale = 'log';
    ax.XLim = [8e-7 1]; 
    ax.YLim = [1e-4 1e-1];
    ax.XTick = logspace(-6,0,4);
    ax.YTick = logspace(-4,-1,4);
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    caxis(ax,[-12 -1]);
    c.Ticks = -11:3:-2;
    c.Label.String = 'log(AE)';
    c.Label.Interpreter = 'latex';
    c.Label.FontSize = 20;
    
    ax.FontName = 'times';
    ax.FontSize = 20;
    ax.LabelFontSizeMultiplier = 24/20;
    ax.TickLabelInterpreter = 'latex';
    ax.XLabel.String = '$y$';
    ax.XLabel.Interpreter = 'latex';
    ax.XLabel.FontSize = 24;
    ax.YLabel.String = '$\epsilon$';
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 24;
    ax.YLabel.Rotation = 0;
    
    prePosition = ax.Position;
    ax.XLabel.Position(1) = 1e-3;
    ax.XLabel.Position(2) = 5e-5;
    ax.YLabel.Position(1) = 7e-8;
    ax.YLabel.Position(2) = 2.3e-3;
    ax.Position  = prePosition;

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