% Start
clc;
close all;
clear all;

   FNC = input('\n  Please enter function number \n  1.PNN without Feature Selection \n  2.GA-PNN with PNN-based fitnessFnc \n  3.GA-ANN with ANN-based fitnessFnc \n  4.GA-PNN with KNN-based fitnessFnc \n  5.GA-ANN with KNN-based fitnessFnc \n  6.PCA-PNN \n  7.Rule Extraction \n  8.Exit \n: ');
   
if FNC == 1
   run('PNN.m');
   elseif FNC == 2
   run('GAPNNPNN.m');
   elseif FNC == 3
   run('GAANNANN.m');
   elseif FNC == 4
   run('GAKNNPNN.m');
   elseif FNC == 5
   run('GAKNNANN.m');
   elseif FNC == 6
   run('PCAPNN.m');  
   elseif FNC == 7
   run('Tree.m');
   elseif FNC == 8
    clc;
    close all;
    disp('logout');
end