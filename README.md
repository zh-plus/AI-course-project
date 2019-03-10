# Course projects from SUSTech CS303 Artificial Intelligence, Fall 2018

*Attention! All the coding is on the basis of given API form course requirements, so the code here are platform specific, which means you cannot take it and apply for your own purpose.*

**Focus on the details of algorithm implementation is more reasonable. These four .pdf reports may help understanding the algorithm and how it is implemented.**

- [Gobang AI project report](https://github.com/zh-plus/AI-course-project/blob/master/Gobang.pdf)
- [CARP project report](https://github.com/zh-plus/AI-course-project/blob/master/CARP.pdf)
- [IMP project report](https://github.com/zh-plus/AI-course-project/blob/master/IMP.pdf)
- [SVM optimization project report](https://github.com/zh-plus/AI-course-project/blob/master/SVM.pdf)



#### Gobang AI project

- Minimax search + Minimax pruning algorithm

*I'm sorry that the structure of this project is terrible. Because I make a mistake in estimating the workload of this project, which make me hurrying up to finish this project. I'll try to refactor it in my spare time.*

No simple usage here, because the code is API specific.



#### Capacitated Arc Routing Problem Project

- Heuristic search + Genetic Algorithm

`$ python3 CARP_solver.py <sample data path> -t <time limit> -s <random seed> `



#### Influence Maximization Problem Project

- Cost-Effective Lazy Forward Selection (CELF)

`$ python3 IMP.py -i <graph description path> -k <the number of seeds> -m <IC/LT> -t <termination time> `



#### Support Vector Machine Project

- Optimize SVM parameters using Sequential Minimal Optimization (SMO)

`$ python3 SVM.py <train data path> <test data path> -t <time limit>`