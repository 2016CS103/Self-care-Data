function Cost = CostFunction_Fcn(X)
NPar = size(X,2);

S1 = 0;
S2= 0;

for ii = 1:NPar
    S1 = S1 + cos(2*X(:,ii));
    S2 = S2 + X(:,ii).^2;    
end
Cost = 10*NPar + S2 - 10 * (S1);
end