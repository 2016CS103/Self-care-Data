function  Best_Featuer =  GAFSPNN(data,PopulationSize,Generations)
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
FitnessFcn = @FitFunc_PNN; 
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
function [Fitness] = FitFunc_PNN(population)
global Data
global GenLen
FeatIndex = find(population==1); %Feature Index
X1 = Data.x;% Features Set
Y1 = grp2idx(Data.y);% Class Information
X1 = X1(:,FeatIndex);
NumFeat = numel(FeatIndex);
    
    
  % Train & Test 
  
    
  for i=1:2
      dsktr.x = X1;
      dsktr.y = Y1;
      kn = size(dsktr.x,1);
      nfold = kn/2 ;    
   %%% dsTrain   
          if (i==1)
           dsktr.x(1:nfold,:) = [];
           dsktr.y(1:nfold,:) = [];
          else
             j = nfold-1;    
             v=i-1 ;
             d = (v*nfold)+1 ;
             h =  d+j ;
             dsktr.x(d:h,:) =[]; 
             dsktr.y(d:h,:) =[];
          end
          
          train.x = dsktr.x ;
          train.y = dsktr.y ;
          
   %%% dstest
         dskts.x = X1;
         dskts.y = Y1;
         if (i==1)
          dskts.x = dskts.x(1:nfold,:);
          dskts.y = dskts.y(1:nfold,:);
         else
          j = nfold-1;    
          v=i-1 ;
          d = (v*nfold)+1 ;
          h =  d+j ;
          dskts.x = dskts.x(d:h,:);  
          dskts.y = dskts.y(d:h,:);
         end
        test.x = dskts.x ;
        test.y = dskts.y ;
        
  %%% Train PNN

        trTc  = rot90(train.y);
        trX   = rot90(train.x);
        T = ind2vec(trTc);
        spread = 1;
        trnet = newpnn(trX,T,spread);
        
  %%% Test PNN

    testY  = rot90(test.y);
    s = length(testY);
    TY = zeros(7,s);
   for m=1:s
      TY(testY(m),m) = 1;
   end
   testX  = rot90(test.x);
   PNNTestOut = sim(trnet,testX);
   %%% Confusionmat
   PNNout = zeros(1,35);
   for p=1:35
       for q=1:size(PNNTestOut,1)
           if (PNNTestOut(q,p)== 1) 
               PNNout(1,p)= q;
           end
       end
   end
   stats = confusionmatStats(testY,PNNout);
   meana = mean(stats.accuracy);
   Fitness = (GenLen-NumFeat)/meana;
  end
end

