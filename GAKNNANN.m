% Start
  clc;
  close all;
  clear all; %#ok<CLALL>
  GArtimes = input('\n Please enter runs tiems: ');
  PopulationSize = input('\n Please enter  GA PopulationSize: ');
  Generations = input('\n Please enter  GA Generations: ');
  Precision = zeros(7,1);
  Recall = zeros(7,1);
  F1score = zeros(7,1);
  
% Dataset
  data = LoadData('Dataset\SCADI.csv');

% GA Featuer Selection



  for r=1:GArtimes 
      fprintf('run time = %i\n',r);
      tic ;
      Best_Featuer{r} = GAFSKNN(data,PopulationSize,Generations);
      disp(' ');
      disp('Elapsed time for GA-FS:');
      toc;
      disp(' ');
  


% ANN


  % DATASET
     
    ds.y = data.y;
    ds.y = grp2idx(data.y);
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
                   
     trn.x = dsktr.x ;
     trn.y = dsktr.y ;
          
          
         
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

        
       
%%% Train ANN

x = trn.x;
T = trn.y;
T = rot90(T);
t = zeros (7,63);
 for f = 1:63
      b = T(1,f);
      t(b,f) = 1 ;
 end
x = rot90(x);

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
[net,tr] = train(net,x,t);

% Test Confusion Plot Variables

test.x = rot90(test.x);
yTst = net(test.x);
tTst = test.y;
tTst = rot90(tTst);
ttTst = zeros (7,7);
 for f = 1:7
      b = tTst(1,f);
      ttTst(b,f) = 1 ;
 end
% Plots      
Ptitle = ['PNN-Fold-' num2str(i) '  ' ];
figure
plotconfusion(ttTst,yTst,'Test');

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

% Stats Variables
stats{r}{i} = confusionmatStats(tTst,ANNtest0);


 end
 end
  
          
          
          
          

