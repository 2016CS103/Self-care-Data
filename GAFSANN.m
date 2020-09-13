function  Best_Featuer =  GAFSANN(data,PopulationSize,Generations)
disp('GA Initialization ...');

global GenLen;
GenLen = data.nx;
global Data
Data = data;

EliteCount = ceil(0.1*PopulationSize) ;
options = gaoptimset('CreationFcn', {@PopFunction},...
                     'PopulationSize',PopulationSize,...
                     'Generations',Generations,...
                     'PopulationType', 'bitstring',... 
                     'SelectionFcn',{@selectionroulette},...
                     'MutationFcn',{@mutationuniform},...
                     'CrossoverFcn', {@crossovertwopoint},...
                     'EliteCount',EliteCount,...
                     'PlotFcns',{@gaplotbestf},...  
                     'Display', 'iter'); 
rng('default');
sd = sum(100*clock);
rng(sd);
nVars = data.nx ; 
FitnessFcn = @FitFunc_ANN; 
[chromosome,~,~,~,~,~] = ga(FitnessFcn,nVars,options);
Best_chromosome = chromosome; % Best Chromosome
Best_Featuer = find(Best_chromosome==1); % Index of Chromosome
end

%%% POPULATION FUNCTION Initial Population
function [population] = PopFunction(GenLen,~,options)
R = rand;  
population = (rand(options.PopulationSize, GenLen)> R);
end

%%% FITNESS FUNCTION  
function [Fitness] = FitFunc_ANN(population)
global Data
global GenLen
FeatIndex = find(population==1); %Feature Index
X1 = Data.x;% Features Set
Y1 = grp2idx(Data.y);% Class Information
X1 = X1(:,FeatIndex);
Target = zeros (7,70);
T = Y1;
T = rot90(T);
 for i = 1:70
      j = T(1,i);
      Target(j,i) = 1 ;
 end
X1 = rot90(X1);
NumFeat = numel(FeatIndex);
net = patternnet(25);
net.divideFcn = 'dividerand';  
net.divideMode = 'sample'; 
net.trainParam.showWindow=false;
net.trainParam.epochs=100;
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net = init(net);

% Train Tool GUI
net.trainParam.showWindow = 0;

% Train the Network
[net,tr] = train(net,X1,Target);

% Test Confusion Plot Variables
yTst = net(X1(:,tr.testInd));
tTst = Target(:,tr.testInd);
ANNout = zeros(1,7);
   for p=1:7
       for q=1:7
           if (tTst(q,p)== 1) 
               ANNout(1,p)= q;
           end
       end
   end
for rx=1:7
    for ry=1:7
        if (yTst(rx,ry) >= 0.9)
            yTst(rx,ry) = 1;
        else
            yTst(rx,ry) = 0;
        end
    end
end

ANNtest0 = NaN(1,7);

for ax = 1:7
    for ay = 1:7
    if (yTst(ay,ax)== 1)
        ANNtest0(1,ax)= ay;
    end
    end
end

for xx = 1:7
    if (ANNtest0(1,xx)== 0);
        ANNtest0(1,xx)= '' ;
    end
end     
stats = confusionmatStats(ANNout,ANNtest0);
meana = mean(stats.accuracy);

Fitness = (GenLen-NumFeat)/meana;
end

