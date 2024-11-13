# A simple implementation of an Interior Point (IP) Method

<img width="399" alt="image" src="https://github.com/user-attachments/assets/e1b0e510-72a5-46fe-a6e9-c238702b8873">

I implemented this Interior Point Method as coursework for the Numerical Optimization Project at the University of Freiburg.
This project accompanies the lecture on Numerical Optimization held by Prof. Dr. Diehl.
Constraining the problem classes to Quadratic Programs (QPs) allows for a more efficient solution of the KKT system through LDLT decompositions.

The implementation follows a modular concept of a numerical solver inspired by [object-oriented software for quadratic-programming (OOQP)](https://pages.cs.wisc.edu/~swright/ooqp/).
The modularity in the first place allows to employ and compare different linear system solvers for the KKT-system.
The above visualization illustrates the interplay of the different modules.

Modified versions of two of the [OSQP problem classes](https://github.com/osqp/osqp_benchmarks) where used to evaluate the implemented method.

### Most urgent TODO's

- [ ] add installation instructions
- [ ] add the report for the project
- [ ] make sure evaluations are comprehensible and reproducible
- [ ] check and improve documentation and type hints

