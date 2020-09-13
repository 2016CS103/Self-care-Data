 % Start
 clc;
 close all;
 clear all; %#ok<*CLSCR> 
 Precision = zeros(7,1);
 Recall = zeros(7,1);
 F1score = zeros(7,1);
 
 % Read Dataset
 filename = 'Dataset\SCADI.csv';
 dataset1 = dataset('file',filename,'Delimiter',',','ReadObsNames','off');
 dataset1 = dataset2cell(dataset1);
 dataset1(1,:) = [];

 % Tree Target
 Treet = dataset1 ;
 Treet = Treet(:,end);
 data.Treet = Treet ;
    
  % Creat TreeAttr
 TreeAttr = dataset1;
 TreeAttr(:,end) = [];
 TreeAttr = cell2mat(TreeAttr);
 
 % Class1->1 ; Class2->2; Class3->3; Class4->4
 % Class5->5 ; Class6->6; Class7->7;
 %
 Treet = grp2idx(Treet);
 
   for i=1:10
      dsktr.x = TreeAttr;
      dsktr.y = Treet;
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
         dskts.x = TreeAttr;
         dskts.y = Treet;
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

 
 % Creat Tree
 DT = fitctree(trn.x,trn.y);
  
 % View Tree
 view(DT);
 view(DT,'mode','graph'); 
 DTFS = DT.CutPredictor ;
 DTFS = rot90(DTFS);
 disp(DTFS);
 
 % Confusionmat
y_hat = predict(DT, test.x);
stats{i} = confusionmatStats(y_hat,test.y);

   end
 
 