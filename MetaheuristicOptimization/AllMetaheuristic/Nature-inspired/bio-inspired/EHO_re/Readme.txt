
Elephant Herding Optimization (EHO)

Gai-Ge Wang
October 12, 2015

email:  gaigewang@163.com
       gaigewang@gmail.com

The files in this zip archive are MATLAB m-files that can be used to study Elephant Herding Optimization (EHO) algorithm.

EHO is the method that we invented and wrote about in the following paper:
Gai-Ge Wang, Suash Deb, Leandro dos Santos Coelho. Elephant Herding Optimization. 
In: 2015 3rd International Symposium on Computational and Business Intelligence (ISCBI 2015), 
Bali, Indonesia, December 7-8, 2015. IEEE, pp ****

Note: 
1) I do not make any effort to rewrite some general codes, while I reuse some codes according to Prof. Dan Simon. We should stand the shoulder of giant, therefore I have more time to focus on our method-EHO. In the following, I will provide the detailed description about all the codes. 
2) The MATLB R2015a is used when implementing our method. 
3) The C++ code of Elephant Herding Optimization (EHO) can be found in the web soon.
4) As discussed in the paper, the EHO like any other metaheuristic algorithms cannot find the best solution for each run. However, our research team will improve it and distribute the codes in our future research. 

The MATLAB files can be used to reproduce the results in the paper, or to do your own experiments. The software is freely available for any purposes (it is on the Internet, after all) although I would of course appreciate an acknowledgement if you use it as part of a paper or presentation.

The MATLAB files and their descriptions are as follows:

Ackley.m: 
This is the benchmark functions discussed in the paper. You can use it as template to write your own function if you are interested in testing or optimizing some other functions. This code is modified according to Dan Simon. The original one is available at http://academic.csuohio.edu/simond/bbo.

EHO_Generation_V1.m, EHO_FEs_V1.m:
Elephant Herding Optimization (EHO) algorithm. The fixed generations (iterations) and fixed Function Evaluations (FEs) are considered as termination condition for EHO_Generation_V1.m and EHO_FEs_V1.m, respectively. It can be used to optimize some function by typing, for example, the following at the MATLAB prompt:
>> EHO_Generation_V1(@Ackley);
This command would run EHO_Generation_V1 on the Ackley function (which is codified in Ackley). 
>> EHO_FEs_V1(@Ackley);
This command would run EHO_FEs_V1 on the Ackley function (which is codified in Ackley). 

Init.m: 
This contains various initialization settings for the optimization methods. You can edit this file to change the population size, the generation count limit, the problem dimension, the maximum Function Evaluations (FEs), and the percentage of population of any of the optimization methods that you want to run. This code is modified according to Dan Simon. The original one is available at http://academic.csuohio.edu/simond/bbo.

ClearDups.m: 
This is used by each optimization method to get rid of duplicate population members and replace them with randomly generated individuals. This code is modified according to Dan Simon. The original one is available at http://academic.csuohio.edu/simond/bbo.

ComputeAveCost.m: 
This is used by each optimization method to compute the average cost of the population and to count the number of legal (feasible) individuals. This code is the same as Dan Simon. The original one are available at http://academic.csuohio.edu/simond/bbo.

PopSort.m: 
This is used by each optimization method to sort population members from most fit to least fit. This code is the same with Dan Simon. The original one is available at http://academic.csuohio.edu/simond/bbo. 

Conclude1.m, Conclude2.m: 
They are concludes the processing of each optimization method. It does common processing like outputting results. Conclude1.m and Conclude2.m are used in EHO_Generation_V1.m and EHO_FEs_V1.m, respectively. They are modified according to Dan Simon. The original one is available at http://academic.csuohio.edu/simond/bbo.

I hope that this software is as interesting and useful to you as is to me. Feel free to contact me with any comments or questions.


