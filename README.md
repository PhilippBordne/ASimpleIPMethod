I implemented this Interior Point Method as coursework for the Numerical Optimization Project at the University of Freiburg.
This project accompanies the lecture on Numerical Optimization held by Prof. Dr. Diehl.

With this implementation I tried to follow a modular concept of a numerical solver inspired by OOQP.
Constraining the problem classes to Quadratic Programs (QPs) allows for a more efficient solution of the KKT system through LDLT decompositions.
