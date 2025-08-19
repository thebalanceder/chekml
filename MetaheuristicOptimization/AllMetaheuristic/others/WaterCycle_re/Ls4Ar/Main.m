clc
clear
%% Import the data set
[filename,filepath]=uigetfile('*.mat','Select Input file');% Select a data set 
addpath(genpath(filepath));%Add the path
load(filename); % Import the data set
rmpath(genpath(filepath));
clear filename filepath
%%
% Use LSAR
% [red,time] = LSAR(C,D,T); 
%
% Use LSAR-ASP
% [red,time] = LSAR_ASP(C,D)
%
%%

