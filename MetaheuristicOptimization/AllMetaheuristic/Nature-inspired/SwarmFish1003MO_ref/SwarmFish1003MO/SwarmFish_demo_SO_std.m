% /*M-FILE SCRIPT SwarmFish_demo_SO_std MMM SGALAB */ %
% /*==================================================================================================
%  Swarm Optimisation and Algorithm Laboratory Toolbox for MATLAB
%
%  Copyright 2012 The SxLAB Family - Yi Chen - leo.chen.yi@live.co.uk
% ====================================================================================================
% File description:
%
%Appendix comments:
%
%Usage:
%
%===================================================================================================
%  See Also:
%
%===================================================================================================
%===================================================================================================
%Revision -
%Date          Name     Description of Change  email
%16-May-2011   Chen Yi  Initial version        leo.chen.yi@live.co.uk
%26-Jul-2011   Chen Yi  Add eachtimeplot       leo.chen.yi@live.co.uk
%12-May-2012   Chen Yi  fprintf( fid,  '\n');  leo.chen.yi@live.co.uk
%HISTORY$
%==================================================================================================*/
%==================================================================================================*/

% SwarmFish_demo_SO_std Begin



% fresh
clear ;
close ('all');
warning off
% to delete old output_*.txt
% !del SwarmFish_O_*.*
delete SwarmFish_O_*.txt
% set working path

% begin to count time during calculating
home ;
tic % timer start >>

% data preparation

% open data files

%%%input data files
fid1  = fopen('SwarmFish_I_min_confines.txt' , 'r' );
fid2  = fopen('SwarmFish_I_max_confines.txt' , 'r' );
fid3  = fopen('SwarmFish_I_try_number.txt' , 'r' );
fid4  = fopen('SwarmFish_I_crowd.txt' , 'r' );
fid5  = fopen('SwarmFish_I_population.txt' , 'r' );
fid6  = fopen('SwarmFish_I_steps.txt' , 'r' );
fid7  = fopen('SwarmFish_I_max_generation.txt' , 'r' );

fid20= fopen('SwarmFish_I_visual.txt','r');

% set total test number
fid1000= fopen('SwarmFish_I_testnumber.txt','r');


%output data files
fid120 = fopen('SwarmFish_O_bestfitness.txt','a+');
fid130 = fopen('SwarmFish_O_maxfitness.txt','a+');
fid140 = fopen('SwarmFish_O_minfitness.txt','a+');
fid150 = fopen('SwarmFish_O_meanfitness.txt','a+');
fid160 = fopen('SwarmFish_O_best_result_space.txt','a+');
%          fid19 = fopen('OUTPUT_best_coding_space.txt','w+');
%          fid20 = fopen('OUTPUT_now_generation.txt','w+');
%          fid21 = fopen('OUTPUT_now_probability_crossover.txt','w+');

% begin to load data from file

% read data from these files
disp('/*==================================================================================================*/')
disp('/*  Swarm Optimisation and Algorithm Laboratory Toolbox 1.0.0.1 */ ')
disp('');
disp('    15-May-2012 Chen Yi leo.chen.yi@live.co.uk Glasgow ')
disp('/*==================================================================================================*/')
disp('>>>>')
disp(' Begin to evaluate...Waiting please ...');

min_confines = fscanf( fid1 , '%g' ); min_confines = min_confines' ;

max_confines = fscanf( fid2 , '%g' ); max_confines = max_confines';
%
%       probability_crossover = fscanf( fid3 , '%g' ); probability_mutation = fscanf(fid4,'%g');
%
population = fscanf( fid5 , '%g' );
%
step = fscanf( fid6 , '%g' );
%
max_generation = fscanf( fid7 , '%g' );
%
trynum = fscanf( fid3 , '%g' );
%
crowd = fscanf( fid4 , '%g' );
%
%       deta_fitness_max = fscanf( fid10 , '%g' );
%
%       max_probability_crossover = fscanf( fid11,'%g' );
%
%       probability_crossover_step = fscanf(fid12,'%g');
%
%       max_no_change_fitness_generation = fscanf(fid13,'%g');
%

%
%       now_probability_crossover = probability_crossover;
%
visual = fscanf(fid20,'%g');
% %

testnumber = fscanf(fid1000,'%g');

disp(' the total test number is');
disp(testnumber);

% Step into SwarmFish()
%
eachtimeplot       = 0;
statisticalplot    = [1,1,1,1,1];
%statisticalplot(1) - plot MIN,MEAN,MAX or not, 0-NO, 1-YES
%statisticalplot(2) - plot mAP or not,          0-NO, 1-YES
%statisticalplot(3) - plot mSTD or not,         0-NO, 1-YES
%statisticalplot(4) - plot mmAP or not,         0-NO, 1-YES
%statisticalplot(5) - plot mmSTD or not,        0-NO, 1-YES

options = { ...
    visual,               % 1, fish visual
    step,                 % 2, fish step
    trynum,               % 3, fish try number,
    crowd,                % 4, fish crowd factor
    eachtimeplot,         % 5, plot each test, 1 - plot each time, 0 - no plots
    statisticalplot    }; % 6, statistical plots, 1 - yes ,  0 - no plots

SwarmFish_Procedure_h = timebar('SwarmsLAB::SwarmFish','Total Progress...');

for idx = 1 : 1 : testnumber
    % Output
    
    timebar( SwarmFish_Procedure_h , idx/testnumber );
    
    disp('Test NO.'); disp(idx);
    disp('');
    
    [   fitness_data ,...
        best_decimal_space ,...
        error_status ]= SwarmFish__entry_SO_std...
        ( options,...
        min_confines ,...
        max_confines ,...
        population ,...
        max_generation,...
        [testnumber,idx] );
    
    disp('');
    
    if ( error_status ~= 0 )  return ;  end
    
    %write data to output files
    
    %   OUTPUT_bestfitness.txt
    fprintf( fid120 , '%f\n', fitness_data(1));
    % add a space to seperate each test loop data
    fprintf(  fid120,  '\n');
    
    %   OUTPUT_maxfitness.txt
    fprintf( fid130 , '%f\n', fitness_data(2));
    % add a space to seperate each test loop data
    fprintf(  fid130,  '\n');
    
    %   OUTPUT_minfitness.txt
    fprintf( fid140,  '%f\n', fitness_data(3));
    % add a space to seperate each test loop data
    fprintf(  fid140,  '\n');
    
    %   OUTPUT_meanfitness.txt
    fprintf(  fid150,  '%f\n', fitness_data(4));
    % add a space to seperate each test loop data
    fprintf(  fid150,  '\n');
    
    %   OUTPUT_best_result_space.txt
    fprintf(  fid160,  '%f\n', best_decimal_space );
    % add a space to seperate each test loop data
    fprintf(  fid160,  '\n');
    
    % %   OUTPUT_best_coding_space
    % fprintf(  fid19 , '%f\n' , best_coding_space );
    
    % %   OUTPUT_now_generation.txt
    % fprintf(  fid20, '%f\n' , now_generation );
    %
    % %   OUTPUT_now_probability_crossover.txt
    % fprintf(  fid21, '%f\n' , now_probability_crossover );
    
    
    
end

%close files
close( SwarmFish_Procedure_h );
status = fclose( 'all' );

disp('End SwarmFish Evaluating');
disp('');

disp(' More detail result in text files with " Swarm_O_*.txt " ' )
disp('----------------------------------------------------------------------------------------')
result_files = list_current_dir_files ('SwarmFish_O*.*')

disp('----------------------------------------------------------------------------------------')

% timer end
toc

clear all
% SwarmFish_demo_SO_std End