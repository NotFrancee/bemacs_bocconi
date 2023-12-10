# Notes on the solutions of the exercises on Greedy and Simulated Annealing

## General notes

It is strongly advised to use [meld](https://meldmerge.org/), or an equivalent program, to look at
the differences between similar files (e.g. between different versions of the same code, like
`"LatinSquare1.py"` and `"LatinSquare2.py"`).

Also, the code is heavily commented. Be sure to read the comments and understand what's going on in
the code and why.

## Associations between the files and the exercises

* The files `"Greedy.py"`, `"SimAnn.py"`, `"TSP.py"` and `"tsprun.py"` are the same files that had
  been uploaded at the end of the lectures.

* Most other files are referred to the Greedy exercises. When unspecified, "Exercise X" means "the
  Exercises X in greedy_exercises.pdf". However, the test scripts all use the Simulated Annealing
  solver.

* The file `"the_wrong_move.py"` is the solution to Exercise 2

* The files `"TSP_SC.py"` and `"tspscrun.py"` are the solution to Exercise 3. The files
  `"TSP_both.py"` and `"tspbothrun.py"` are the same solution, but to the more advanced version
  which uses inheritance and implements both move schemes (cross-links and swap-cities). Also,
  `"tspbothrun.py"` tests both schemes on the same problem instance, using the code in
  `"TSP_both.py"`.

* The files `"LatinSquare1.py"` and `"lsq1run.py"` are the solution to Exercise 4. The solutions to
  Exercises 5 and 6 are called the same, with "2" and "3" in the names instead of "1". Each version
  builds on the previous one. Version "3" is basically a one-line modification of version "2". The
  directory `"verymuchadvanced"` contains a solution to the puzzle in Exercise 4 (although it's
  actually a modification of the solution of Exercise 6, it can be used in all exercises from 4 up
  to 9); it's only intended for hardcore enthusiasts.

* The files `"Sudoku.py"` and `"sudokurun.py"` are the solutions to Exercise 7. The Sudoku class is
  derived from the one in `"LatinSquare3.py"`, as suggested in the exercise. Additionally, the
  `__repr__` method was rewritten to make it nicer, but this is largely unnecessary.

* The files `"Sudoku_Solver1.py"` and `"sudokusolver1run.py"` are the solution to Exercise 8. Same
  for Exercise 9, with the change "1" → "2" in the names.

* The files `"MaxCut.py"` and `"maxcutrun.py"` are the solutions to Exercise 10.

* The files `"MagicSquares.py"` and `"magicsquarerun.py"` are the solutions to Exercise 11.

* Exercises 3 and 4 of "simann_exercises.pdf" are not really exercises with a proper solution.
  Anyway, an example of what one could do is in the sub-directory `"enhanced"`, where there is an
  enhanced version of `"SimAnn.py"` with some additional comments. In that directory, there is a
  file `"Sudoku.py"` which is just a copy of the previous `"Sudoku_Solver2.py"`, and a file
  `"sudokurun.py"` which is basically the same as `"sudokusolver2run.py"` but using hooks to exit
  early in case a solution is found.
