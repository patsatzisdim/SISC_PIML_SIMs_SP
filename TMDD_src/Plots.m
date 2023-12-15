clear
clc

%% PLOTTING SCRIPT
fPlot = 1;      % 1: for the SIM
                % 2-5: error plots of QSSA, GSPT Oe1, GSPT Oe2, CSP
                % 6-7: error plots of SLFNN, RPNN

if fPlot == 1
    %% load testing data that are on SIM
    inSz = 3;           % number of slow variables + 1
    outSz = 1;          % number of fast variables
    nSamples = 100;
    nICs = 10;
    load TMDDTestEps1e-2 allData;
    Ydata = allData(2:3,:);
    Epsdata = allData(4,:);
    Xdata = allData(1,:);
    
    %% form matrices, each column is one trajectory
    y_grid = reshape(Ydata(1,:),[nSamples nICs^(inSz-1)]);
    z_grid = reshape(Ydata(2,:),[nSamples nICs^(inSz-1)]);
    x_grid = reshape(Xdata(1,:),[nSamples nICs^(inSz-1)]);

    %% show some trajectories approaching the SIM
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
    eps = 1e-2;                   % select this epsilon to generate trajectories
    parVec = [k1 k2 k3 k4 eps];
    % all points of ode15s, no cut of transient, no Domain Cut, but not only on the SIM trajectory
    y_init1 = [4; 1.8; 1.6];
    [yAll1, tSol1] = getTMDDdata(parVec,inSz,0,y_init1,false,0);  
    y_init2 = [3.; 1.3; 2.4];
    [yAll2, tSol2] = getTMDDdata(parVec,inSz,0,y_init2,false,0);


    figure(1);
    ax = axes('OuterPosition',[0 0 1 1]);
    surf(ax,y_grid,z_grid,x_grid,x_grid,'EdgeColor','interp','FaceColor','interp'); hold on;
    scatter3(ax,yAll1(2,1),yAll1(3,1),yAll1(1,1)+0.01,100,'rs','filled'); hold on;
    plot3(ax,yAll1(2,:),yAll1(3,:),yAll1(1,:)+0.01,'r-','LineWidth',2); hold on;
    scatter3(ax,yAll2(2,1),yAll2(3,1),yAll2(1,1)+0.01,100,'rs','filled'); hold on;
    plot3(ax,yAll2(2,:),yAll2(3,:),yAll2(1,:)+0.01,'r-','LineWidth',2); hold off;
    ax.XLim = [0.15 2.1];
    ax.YLim = [1.4 2.7];
    ax.ZLim = [0.2 5.5];
    ax.XTick = linspace(0.5,2.,4);
    ax.YTick = linspace(1.4,2.6,4);
    ax.ZTick = linspace(1,5,5);  
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.ZMinorTick = 'on';
    view(140,20);

    ax.FontName = 'times';
    ax.FontSize = 20;
    ax.LabelFontSizeMultiplier = 24/20;
    ax.TickLabelInterpreter = 'latex';
    ax.XLabel.String = '$y$';
    ax.XLabel.Interpreter = 'latex';
    ax.XLabel.FontSize = 24;
    ax.YLabel.String = '$z$';
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 24;
    ax.ZLabel.String = '$x$';
    ax.ZLabel.Interpreter = 'latex';
    ax.ZLabel.FontSize = 24;
    ax.ZLabel.Rotation = 0;
    ytickangle(0);

    prePosition = ax.Position;
    ax.Position  = prePosition;

    return

elseif (fPlot >= 2) && (fPlot<=5)
    %% load testing data
    inSz = 3;   % number of slow variables + 1
    outSz = 1;  % number of fast variables
    nSamples = 100;
    nICs = 5;
    load TMDDTest allData;
    Ydata = allData(2:3,:);
    Epsdata = allData(4,:);
    Xdata = allData(1,:);
    
    %% params for GSPT approximations
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

    if fPlot == 2
        % oe0
        QSSAo0SIM = (k1*Ydata(2,:)+1)./Ydata(1,:);   % QSSA: x = k1*z+1/y 
        AErr = abs(Xdata-QSSAo0SIM);
    elseif fPlot == 3
        % oe0
        QSSAo0SIM = (k1*Ydata(2,:)+1)./Ydata(1,:);   % QSSA: x = k1*z+1/y
        % oe1
        GSPT_o1C = -((k3 + k1*k3*Ydata(2,:) + Ydata(1,:).*(k2 + k1*k2 + k4 + k1*(-1 +...
        k2 + k4).*Ydata(2,:)))./Ydata(1,:).^3);
        o1_GSPT = QSSAo0SIM + Epsdata.*GSPT_o1C;
        AErr = abs(Xdata-o1_GSPT);
    elseif fPlot == 4  
        % oe0
        QSSAo0SIM = (k1*Ydata(2,:)+1)./Ydata(1,:);   % QSSA: x = k1*z+1/y
        % oe1
        GSPT_o1C = -((k3 + k1*k3*Ydata(2,:) + Ydata(1,:).*(k2 + k1*k2 + k4 + k1*(-1 +...
        k2 + k4).*Ydata(2,:)))./Ydata(1,:).^3);
        % oe2
        GSPT_o2C = (k3^2*(1 + k1*Ydata(2,:)).*(4 + k1*Ydata(2,:)) + Ydata(1,:).^2.*((k2 + k4)*(k2 + 2*k4) + ...
           k1*k2*(-1 + 3*k2 + 4*k4) + k1*(-1 + k2 + k4)*(-1 + k2 + 2*k4)*Ydata(2,:) + ...
           k1^2*k2*(k2 + (-1 + k2 + k4)*Ydata(2,:))) + k3*Ydata(1,:).*(6*k4 + k1*Ydata(2,:).*(-4 + ...
           7*k4 + k1*(-1 + k4)*Ydata(2,:)) + k2*(4 + k1*(5 + Ydata(2,:).*(5 + k1*(2 + Ydata(2,:)))))))./Ydata(1,:).^5;
        o2_GSPT = QSSAo0SIM + Epsdata.*GSPT_o1C + Epsdata.^2.*GSPT_o2C;
        AErr = abs(Xdata-o2_GSPT);
    elseif fPlot == 5
        % CSP with one iteration SIM
        CSPo2 = -(Epsdata(1,:).^2*k2^2 + Epsdata(1,:).*((2 + k1)*k2 + k4).*Ydata(1,:) + Ydata(1,:).^2 - ...
         Epsdata(1,:)*k1*k3.*Ydata(2,:) - sqrt((Epsdata(1,:).^2*k2^2 + Epsdata(1,:).*((2 + k1)*k2 + ...
         k4).*Ydata(1,:) + Ydata(1,:).^2 - Epsdata(1,:)*k1*k3.*Ydata(2,:)).^2 + ...
         4*Epsdata(1,:)*k3.*Ydata(1,:).*(Epsdata(1,:)*k2 + Ydata(1,:) + k1*(Epsdata(1,:)*(1 + ...
         k2 + k1*k2) + Ydata(1,:)).*Ydata(2,:))))./(2.*Epsdata(1,:)*k3.*Ydata(1,:));
        AErr = abs(Xdata-CSPo2);
    end

    %%
    figure(1);
    ax = axes('OuterPosition',[0 0 1 1]);
    scatter3(ax,Ydata(1,:),Ydata(2,:),Epsdata,50,log10(AErr),'.');
    c = colorbar;
    colormap(jet)
    ax.ZScale = 'log';
    ax.XLim = [0 2.]; 
    ax.YLim = [1.4 3.0];
    ax.ZLim = [1e-4 1e-1];
    ax.XTick = linspace(0,2,3);
    ax.YTick = linspace(1.5,3,4);
    ax.ZTick = logspace(-4,-1,4);
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    caxis(ax,[-9 0]);
    c.Ticks = -8:2:0;
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
    ax.YLabel.String = '$z$';
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 24;
    ax.ZLabel.String = '$\epsilon$';
    ax.ZLabel.Interpreter = 'latex';
    ax.ZLabel.FontSize = 24;
    ax.ZLabel.Rotation = 0;
    ytickangle(0);

    view(-40,25);

    prePosition = ax.Position;
    ax.XLabel.Position(1) = 1.4;
    ax.XLabel.Position(2) = 1.1;
    ax.XLabel.Position(3) = 8e-5;
    ax.YLabel.Position(1) = 0.05;
    ax.YLabel.Position(2) = 2.7;
    ax.YLabel.Position(3) = 2e-5;
    ax.ZLabel.Position(1) = -0.43;
    ax.ZLabel.Position(3) = 0.002;
    ax.Position  = prePosition;

    return

elseif (fPlot>=6) && (fPlot<=7)
    %% load testing data
    inSz = 3;   % number of slow variables + 1
    outSz = 1;  % number of fast variables
    nSamples = 100;
    nICs = 5;
    load TMDDTest allData;
    Ydata = allData(2:3,:);
    Epsdata = allData(4,:);
    Xdata = allData(1,:);

    if fPlot == 6
        load learned_PI_SLFNN learned;
        netDim = [3 1 20 1];
        preds = forwardNN([Ydata; Epsdata],learned,netDim);
        AErr = abs(Xdata-preds);
    elseif fPlot == 7
        load learned_PI_RPNN learned;
        N = 400;
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
    scatter3(ax,Ydata(1,:),Ydata(2,:),Epsdata,50,log10(AErr),'.');
    c = colorbar;
    colormap(jet)
    ax.ZScale = 'log';
    ax.XLim = [0 2.]; 
    ax.YLim = [1.4 3.0];
    ax.ZLim = [1e-4 1e-1];
    ax.XTick = linspace(0,2,3);
    ax.YTick = linspace(1.5,3,4);
    ax.ZTick = logspace(-4,-1,4);
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    caxis(ax,[-9 0]);
    c.Ticks = -8:2:0;
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
    ax.YLabel.String = '$z$';
    ax.YLabel.Interpreter = 'latex';
    ax.YLabel.FontSize = 24;
    ax.ZLabel.String = '$\epsilon$';
    ax.ZLabel.Interpreter = 'latex';
    ax.ZLabel.FontSize = 24;
    ax.ZLabel.Rotation = 0;
    ytickangle(0);

    view(-40,25);

    prePosition = ax.Position;
    ax.XLabel.Position(1) = 1.4;
    ax.XLabel.Position(2) = 1.1;
    ax.XLabel.Position(3) = 8e-5;
    ax.YLabel.Position(1) = 0.05;
    ax.YLabel.Position(2) = 2.7;
    ax.YLabel.Position(3) = 2e-5;
    ax.ZLabel.Position(1) = -0.43;
    ax.ZLabel.Position(3) = 0.002;
    ax.Position  = prePosition;

    return
    
end



%% Functions to predict NN

%%%%%%%%%%%%  FOR PHYSICS INFORMED SLFNN %%%%%%%%%%%%%%%%%%%
%
%
% function for calculating NN output
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

end

%% activation function
%
% seperate so you can change it wherever
function s=actFun(x)
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