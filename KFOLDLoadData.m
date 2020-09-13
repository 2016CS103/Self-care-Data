function data = KFOLDLoadData(filename,i,k)
          datasetk1 = dataset('file',filename,'Delimiter',',','ReadObsNames','off');
          datasetk1 = dataset2cell(datasetk1);
          datasetk1(1,:) = [];
          kn = size(datasetk1,1);
          nfold = kn/k ;
          
          
  %%%%%%% Train   
          dataset1kfold = datasetk1 ;
          if (i==1)
          dataset1kfold(1:nfold,:) = [];
          else
          j = nfold-1;    
          v=i-1 ;
          d = (v*nfold)+1 ;
          h =  d+j ;
          dataset1kfold(d:h,:) =[]; 
          end
          
          data.trx = dataset1kfold ;
          data.try = dataset1kfold ;
          data.trx(:,end) = [];
          data.trx = cell2mat(data.trx);
          data.try(:,1:end-1) = [];  
          data.ntrx = size(data.trx,2);
          
  %%%%%%% Test
          dataset1kfold = datasetk1;
          if (i==1)
          dataset1kfold = dataset1kfold(1:nfold,:);
          else
          j = nfold-1;    
          v=i-1 ;
          d = (v*nfold)+1 ;
          h =  d+j ;
          dataset1kfold = dataset1kfold(d:h,:);   
          end
          
          data.testx = dataset1kfold ;
          data.testy = dataset1kfold ;
          data.testx(:,end) = [];
          data.testx = cell2mat(data.testx);
          data.testy(:,1:end-1) = [];  
          data.ntestx = size(data.testx,2);        
          
end