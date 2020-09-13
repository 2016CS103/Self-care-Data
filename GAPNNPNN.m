% Start
  clc;
  close all;
  clear all; %#ok<CLALL>
  GArtimes = input('\n Please enter runs tiems: ');
  PopulationSize = input('\n Please enter  GA PopulationSize: ');
  Generations = input('\n Please enter  GA Generations: ');


% Dataset
  data = LoadData('Dataset\SCADI.csv');

% GA Featuer Selection

  

  for r=1:GArtimes 
      fprintf('run time = %i\n',r);
      tic ;
      Best_Featuer{r} = GAFSPNN(data,PopulationSize,Generations);
      disp(' ');
      disp('Elapsed time for GA-FS:');
      toc;
      disp(' ');
  
 
% PNN

  % DATASET
     
    ds.y = data.y;
    ds.y(strcmp('class1', ds.y)) = {1};
    ds.y(strcmp('class2', ds.y)) = {2};
    ds.y(strcmp('class3', ds.y)) = {3};
    ds.y(strcmp('class4', ds.y)) = {4};
    ds.y(strcmp('class5', ds.y)) = {5};
    ds.y(strcmp('class6', ds.y)) = {6};
    ds.y(strcmp('class7', ds.y)) = {7};
 
    dsxsize1 = size(data.x,1);
    dsxsize2 = size(Best_Featuer{r},2);
    ds.x = zeros(dsxsize1,dsxsize2);
    
    for i=1:dsxsize1
        for j=1:dsxsize2
            k = Best_Featuer{r}(j);
            ds.x(i,j)= data.x(i,k);
        end    
        
    end  
    
    
  % Train & Test 
  
    
  for i=1:10
      dsktr.x = ds.x;
      dsktr.y = ds.y;
      kn = size(dsktr.x,1);
      nfold = kn/10 ;    
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
         dskts.x = ds.x;
         dskts.y = ds.y;
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
        trTc  = cell2mat(trTc);
        T = ind2vec(trTc);
        spread = 1;
        trnet = newpnn(trX,T,spread);
        Y = sim(trnet,trX);
        
  %%% Test PNN

    testY  = rot90(test.y);
    testY  = cell2mat(testY);
    s = length(testY);
    TY = zeros(7,s);
   for m=1:s
      TY(testY(m),m) = 1;
   end
   testX  = rot90(test.x);
   PNNTestOut = sim(trnet,testX);
   
   %%% Confusionmat
   PNNout = zeros(1,7);
   for p=1:7
       for q=1:7
           if (PNNTestOut(q,p)== 1) 
               PNNout(1,p)= q;
           end
       end
   end
   n = 1;
   Ptitle = ['PNN-Fold-' num2str(r) '-' num2str(i*n) '  ' ];
   figure
   plotconfusion(TY,PNNTestOut,Ptitle);
   stats{r}{i} = confusionmatStats(testY,PNNout);
   n = n +1;
  end  
  end
  

          
          
          
          
          

