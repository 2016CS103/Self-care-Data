function  Best_Featuer =  GAFSKNN(data,PopulationSize,Generations)
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
FitnessFcn = @FitFunc_KNN; 
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
function [Fitness] = FitFunc_KNN(population)
global Data
global GenLen
FeatIndex = find(population==1); %Feature Index
X1 = Data.x;% Features Set
Y1 = grp2idx(Data.y);% Class Information
X1 = X1(:,FeatIndex);
NumFeat = numel(FeatIndex);
Compute = fitcknn(X1,Y1,...
   'NumNeighbors',3,...
   'NSMethod','exhaustive',...
   'Distance','euclidean'); 
Fitness = resubLoss(Compute)/(GenLen-NumFeat);
end

