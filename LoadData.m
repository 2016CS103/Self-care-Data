function data = LoadData(filename)

 % Read Dataset
 ds = dataset('file',filename,'Delimiter',',','ReadObsNames','off');
 ds = dataset2cell(ds);
 ds(1,:) = [];

 % Tree Target
 Treet = ds ;
 Treet = Treet(:,end);
 data.Treet = Treet ;

   
 data.x = ds ;
 data.y = ds ;
 data.x(:,end) = [];
 data.x = cell2mat(data.x);
 data.y(:,1:end-1) = [];  
 data.nx = size(data.x,2);    
end
