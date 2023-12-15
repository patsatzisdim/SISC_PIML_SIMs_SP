clear
clc

%% PLOTTING SCRIPT
fPlot = 1;      % 1: for the SIM
                % 2-5: error plots of QSSA, GSPT Oe1, GSPT Oe2, CSP
                % 6-7: error plots of SLFNN, RPNN

if fPlot == 1
    %% load testing data that are on SIM
    inSz = 3;   % number of slow variables + 1
    outSz = 1;  % number of fast variables
    nSamples = 400;
    nICs = 10;
    load TLCTestEps1e-2 allData;
    Ydata = allData(2:3,:);
    Epsdata = allData(4,:);
    Xdata = allData(1,:);
    
    %% use delauney triangles to construct the surface
    tri = delaunay(Ydata(1,:),Ydata(2,:));

    %% show some trajectories approaching the SIM
    a = 0.1;
    b = 0.6;
    k = 1;
    eps = 1e-2;                   % select this epsilon to generate trajectories

    parVec = [a b k eps];
    y_init1 = [1.; .55; 0.6];
    % all points of ode15s, no cut of transient, no Domain Cut, but not only on the SIM trajectory
    [yAll1, tSol1] = getTLCdata(parVec,inSz,0,y_init1,false,0);  
    y_init2 = [1.2; 0.8; 1.];
    [yAll2, tSol2] = getTLCdata(parVec,inSz,0,y_init2,false,0);


    figure(1);
    ax = axes('OuterPosition',[0 0 1 1]);
    trisurf(tri,Ydata(1,:),Ydata(2,:),Xdata(1,:),'EdgeColor','interp','FaceColor','interp'); hold on;
    scatter3(ax,yAll1(2,1),yAll1(3,1),yAll1(1,1)+0.01,100,'rs','filled'); hold on;
    plot3(ax,yAll1(2,1:end),yAll1(3,1:end),yAll1(1,1:end)+0.01,'r-','LineWidth',2); hold on;
    scatter3(ax,yAll2(2,1),yAll2(3,1),yAll2(1,1)+0.01,100,'ks','filled'); hold on;
    plot3(ax,yAll2(2,:),yAll2(3,:),yAll2(1,:)+0.01,'k-','LineWidth',2); hold off;
    ax.XLim = [0.25 1.4];
    ax.YLim = [0.3 2.1];
    ax.ZLim = [0.2 1.5];
    ax.XTick = linspace(0.4,1.3,4);
    ax.YTick = linspace(0.6,1.8,4);
    ax.ZTick = linspace(0.4,1.3,4);  
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.ZMinorTick = 'on';
    view(-120,25);

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
    ytickangle(0); xtickangle(0);

    prePosition = ax.Position;
    ax.XLabel.Position(1) = 1.2;
    ax.XLabel.Position(2) = 2.2;
    ax.XLabel.Position(3) = 0.0;
    ax.YLabel.Position(1) = 0.2;
    ax.YLabel.Position(2) = 0.7;
    ax.YLabel.Position(3) = 0.06;
    ax.ZLabel.Position(1) = 1.53;
    ax.ZLabel.Position(2) = 2.4;
    ax.ZLabel.Position(3) = 0.8;
    ax.Position  = prePosition;

    return

elseif (fPlot >= 2) && (fPlot<=5)
    %% load testing data
    inSz = 3;   % number of slow variables + 1
    outSz = 1;  % number of fast variables
    nSamples = 100;
    nICs = 5;
    load TLCTest allData;
    Ydata = allData(2:3,:);
    Epsdata = allData(4,:);
    Xdata = allData(1,:);
    
    %% params for GSPT approximations
    a = 0.1;
    b = 0.6;
    k = 1;

    if fPlot == 2
        % oe0
        QSSAo0SIM = Ydata(1,:).*Ydata(2,:)/k;
        AErr = abs(Xdata-QSSAo0SIM);
    elseif fPlot == 3
        % oe0
        QSSAo0SIM =  Ydata(1,:).*Ydata(2,:)/k;   % QSSA: x = k1*z+1/y
        % oe1
        GSPT_o1C = (-(b*Ydata(1,:)) + Ydata(2,:).*(Ydata(1,:).*(1 + a + Ydata(1,:).^2) - ...
            (a + Ydata(1,:).^2).*Ydata(2,:)))./(k^2*Ydata(1,:));
        o1_GSPT = QSSAo0SIM + Epsdata.*GSPT_o1C;
        AErr = abs(Xdata-o1_GSPT);
    elseif fPlot == 4  
        % oe0
        QSSAo0SIM =  Ydata(1,:).*Ydata(2,:)/k;   % QSSA: x = k1*z+1/y
        % oe1
        GSPT_o1C = (-(b*Ydata(1,:)) + Ydata(2,:).*(Ydata(1,:).*(1 + a + Ydata(1,:).^2) - ...
            (a + Ydata(1,:).^2).*Ydata(2,:)))./(k^2*Ydata(1,:));
        % oe2
        GSPT_o2C = (b*Ydata(1,:).*(-(Ydata(1,:).*(1 + a + Ydata(1,:).^2)) + 2*(a + Ydata(1,:).^2).*Ydata(2,:)) + ... 
           Ydata(2,:).*(a*Ydata(1,:).*(Ydata(1,:) + 2*Ydata(1,:).^3 + Ydata(2,:) - 6*Ydata(1,:).^2.*Ydata(2,:)) + ... 
           a^2*(Ydata(1,:).^2 - 2*Ydata(1,:).*Ydata(2,:) - Ydata(2,:).^2) + ...
           Ydata(1,:).^3.*(-2*Ydata(2,:) + Ydata(1,:).*(3 + Ydata(1,:).^2 - 4*Ydata(1,:).*Ydata(2,:) + ...
           Ydata(2,:).^2))))./(k^3*Ydata(1,:).^3);
        o2_GSPT = QSSAo0SIM + Epsdata.*GSPT_o1C + Epsdata.^2.*GSPT_o2C;
        AErr = abs(Xdata-o3_GSPT);
    elseif fPlot == 5
        % CSP with one iteration
        CSPo2 = ((Epsdata(1,:)*k.*Ydata(1,:) + k^2*Ydata(1,:).^2 - a*Epsdata(1,:)*k.*Ydata(2,:) + ...
        2*Epsdata(1,:).^2.*Ydata(1,:).*Ydata(2,:) - Epsdata(1,:)*k.*Ydata(1,:).^2.*Ydata(2,:)).*(1 - ...
        sqrt(1 - (4*Epsdata(1,:).^2*k.*(-(b*Epsdata(1,:).*Ydata(1,:).^2) + 2*Epsdata(1,:).*Ydata(1,:).^2.*Ydata(2,:) + ...
        a*Epsdata(1,:).*Ydata(1,:).^2.*Ydata(2,:) + k*Ydata(1,:).^3.*Ydata(2,:) + Epsdata(1,:).*Ydata(1,:).^4.*Ydata(2,:) - ... 
        2*a*Epsdata(1,:).*Ydata(1,:).*Ydata(2,:).^2 - 2*Epsdata(1,:).*Ydata(1,:).^3.*Ydata(2,:).^2))./(Epsdata(1,:)*k.*Ydata(1,:) + ...
        k^2.*Ydata(1,:).^2 - a*Epsdata(1,:)*k.*Ydata(2,:) + 2*Epsdata(1,:).^2.*Ydata(1,:).*Ydata(2,:) - ...
        Epsdata(1,:)*k.*Ydata(1,:).^2.*Ydata(2,:)).^2)))./(2.*Epsdata(1,:).^2*k);
        AErr = abs(Xdata-CSPo2);
    end

    %%
    figure(1);
    ax = axes('OuterPosition',[0 0 1 1]);
    scatter3(ax,Ydata(1,:),Ydata(2,:),Epsdata,50,log10(AErr),'.');
    c = colorbar;
    colormap(jet)
    ax.ZScale = 'log';
    ax.XLim = [0.2 1.5]; 
    ax.YLim = [0.2 2.2];
    ax.ZLim = [1e-4 1e-1];
    ax.XTick = linspace(0.2,1.4,4);
    ax.YTick = linspace(0.5,2.,4);
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
    ax.XLabel.Position(1) = 1.35;
    ax.XLabel.Position(2) = 0.2;
    ax.XLabel.Position(3) = 3e-5;
    ax.YLabel.Position(1) = 0.05;
    ax.YLabel.Position(2) = 1.4;
    ax.YLabel.Position(3) = 4e-5;
    ax.ZLabel.Position(1) = -0.08;
    ax.ZLabel.Position(3) = 0.002;
    ax.Position  = prePosition;

    return

elseif (fPlot>=6) && (fPlot<=7)
    %% load testing data
    inSz = 3;   % number of slow variables + 1
    outSz = 1;  % number of fast variables
    nSamples = 100;
    nICs = 5;
    load TLCTest allData;
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
        N = 101;
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
    ax.XLim = [0.2 1.5]; 
    ax.YLim = [0.2 2.2];
    ax.ZLim = [1e-4 1e-1];
    ax.XTick = linspace(0.2,1.4,4);
    ax.YTick = linspace(0.5,2.,4);
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
    ax.XLabel.Position(1) = 1.35;
    ax.XLabel.Position(2) = 0.2;
    ax.XLabel.Position(3) = 3e-5;
    ax.YLabel.Position(1) = 0.05;
    ax.YLabel.Position(2) = 1.4;
    ax.YLabel.Position(3) = 4e-5;
    ax.ZLabel.Position(1) = -0.08;
    ax.ZLabel.Position(3) = 0.002;
    ax.Position  = prePosition;

    return
    
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