# Course projects from SUSTech CS303 Artificial Intelligence, Fall 2018

Textbook: [Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu/)



*Attention! All the coding is on the basis of given API form course requirements, so the code here are platform specific, which means you cannot take it and apply for your own purpose.*

**Focus on the details of algorithm implementation is more reasonable. These four .pdf reports may help understanding the algorithm and how it is implemented.**

- [Gobang AI project report](https://github.com/zh-plus/AI-course-project/blob/master/Gobang.pdf)
- [CARP project report](https://github.com/zh-plus/AI-course-project/blob/master/CARP.pdf)
- [IMP project report](https://github.com/zh-plus/AI-course-project/blob/master/IMP.pdf)
- [SVM optimization project report](https://github.com/zh-plus/AI-course-project/blob/master/SVM.pdf)



### Gobang AI project

Algorithm: Minimax search + Minimax pruning algorithm

*I'm sorry that the structure of this project is terrible. Because I make a mistake in estimating the workload of this project, which make me hurrying up to finish this project. I'll try to refactor it in my spare time.*

Usage: No simple usage here, because the code is API specific.

###### ref:

[*Searching for Solutions in Games and Artificial Intelligence*](http://www.dphu.org/uploads/attachements/books/books_3721_0.pdf) 



### Capacitated Arc Routing Problem Project

Algorithm: Heuristic search + Genetic Algorithm

Usage: `$ python3 CARP_solver.py <sample data path> -t <time limit> -s <random seed> `

###### ref:

[*Memetic algorithm with extended neighborhood search for capacitated arc routing problems*](https://ieeexplore.ieee.org/document/5200351)



### Influence Maximization Problem Project

- Cost-Effective Lazy Forward Selection (CELF)

`$ python3 IMP.py -i <graph description path> -k <the number of seeds> -m <IC/LT> -t <termination time> `

###### ref:

[*Scalable influence maxi-mization for prevalent viral marketing in large-scale social networks*](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/msr-tr-2010-2_v2.pdf)

[*Maximizing the spread of influence through a social network*](https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf)

[*Cost-effective outbreak detection in networks*](https://www.cs.cmu.edu/~jure/pubs/detect-kdd07.pdf)



### Support Vector Machine Project

- Optimize SVM parameters using Sequential Minimal Optimization (SMO)

`$ python3 SVM.py <train data path> <test data path> -t <time limit>`

###### ref:

[The Simplified SMO Algorithm](http://cs229.stanford.edu/materials/smo.pdf)

[Sequential Minimal Optimization for SVM](http://web.cs.iastate.edu/honavar/smo-svm.pdf)

[ ”Machine Learning”](https://www.amazon.cn/dp/B01ARKEV1G)
