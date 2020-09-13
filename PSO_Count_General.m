%% Start of PSO
clc
clear
close all

%% Problem Statement
NPar = 3;
VarMin = [-15 -10 -12];
VarMax = [25 5 5];
CostFuncName = 'CostFunction_Fcn';

%% Algorithm's Parameters
SwarmSize = 70;
MaxIteration = 70;
C1 = 2; % Cognition Coefficient;
C2 = 4 - C1; % Social Coefficient;
%% Initial Population
GBest.Cost = inf;
GBest.Position = [];
GBest.CostMAT = [];
for p = 1:SwarmSize
    Particle(p).Position = rand(1,NPar) .* (VarMax - VarMin) + VarMin;
    Particle(p).Cost = feval(CostFuncName,Particle(p).Position);
    Particle(p).Velocity = [];
    Particle(p).LBest.Position = Particle(p).Position;
    Particle(p).LBest.Cost = Particle(p).Cost;
    
    if Particle(p).LBest.Cost < GBest.Cost
        GBest.Cost = Particle(p).LBest.Cost;
        GBest.Position = Particle(p).LBest.Position;
    end
end

%% Start of Optimization
for Iter = 1:MaxIteration
    %% Velocity update
    for p = 1:SwarmSize
        Particle(p).Velocity = C1 * rand * (Particle(p).LBest.Position - Particle(p).Position) + C2 * rand * (GBest.Position - Particle(p).Position);
        Particle(p).Position = Particle(p).Position + Particle(p).Velocity;
        
        Particle(p).Position = max(Particle(p).Position , VarMin);
        Particle(p).Position = min(Particle(p).Position , VarMax);        
        
        Particle(p).Cost = feval(CostFuncName,Particle(p).Position);
        
        if Particle(p).Cost < Particle(p).LBest.Cost
            Particle(p).LBest.Position = Particle(p).Position;
            Particle(p).LBest.Cost = Particle(p).Cost;
            
            if Particle(p).LBest.Cost < GBest.Cost
                GBest.Cost = Particle(p).LBest.Cost;
                GBest.Position = Particle(p).LBest.Position;
            end
        end
    end
    %% Display
    disp(['Itretion = ' num2str(Iter) '; Best Cost = ' num2str(GBest.Cost) ';'])
    GBest.CostMAT = [GBest.CostMAT GBest.Cost];
end
GBest.Position
plot(GBest.CostMAT)