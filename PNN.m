% Start
clc;
close all;
clear all;

% Dataset
for p=1:10
% Dataset
data = KFOLDLoadData('Dataset\SCADI.csv',p,10);

trX   = data.trx;
trTc  = data.try;
testX = data.testx;
testY = data.testy;

 % Class1->1 ; Class2->2; Class3->3; Class4->4 Class5->5 Class6->6 Class7->7
 trTc(strcmp('class1', trTc)) = {1};
 trTc(strcmp('class2', trTc)) = {2};
 trTc(strcmp('class3', trTc)) = {3};
 trTc(strcmp('class4', trTc)) = {4};
 trTc(strcmp('class5', trTc)) = {5};
 trTc(strcmp('class6', trTc)) = {6};
 trTc(strcmp('class7', trTc)) = {7};
  
 testY(strcmp('class1', testY)) = {1};
 testY(strcmp('class2', testY)) = {2};
 testY(strcmp('class3', testY)) = {3};
 testY(strcmp('class4', testY)) = {4};
 testY(strcmp('class5', testY)) = {5};
 testY(strcmp('class6', testY)) = {6};
 testY(strcmp('class7', testY)) = {7};

% Train PNN

trTc  = rot90(trTc);
trX   = rot90(trX);
trTc  = cell2mat(trTc);

T = ind2vec(trTc);
spread = 1;
trnet = newpnn(trX,T,spread);

Y = sim(trnet,trX);

% Test PNN

testY  = rot90(testY);
testY  = cell2mat(testY);
s = length(testY);
TY = zeros(7,s);
for i=1:s
    TY(testY(i),i) = 1;
end
    
testX  = rot90(testX);
TestOut = sim(trnet,testX);

Ptitle = ['PNN-Fold-' num2str(p) '  ' ];
figure
plotconfusion(TY,TestOut,Ptitle);
PNNout = zeros(1,7);
   for pq=1:7
       for q=1:7
           if (TestOut(q,pq)== 1) 
               PNNout(1,pq)= q;
           end
       end
   end
   stats{p} = confusionmatStats(testY,PNNout);
end