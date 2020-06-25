```
==========================
  How to install the program
==========================
 The following command compiles the program in your Linux computer:
 
      g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp
 
  The '-static' flag should be removed on Mac OS, which does not support
  building static executables.
 
  ======================
  How to use the program
  ======================
  You can run the program without argument to obtain the document.
  Briefly, you can compare two structures by:
 
      ./TMscore structure1.pdb structure2.pdb
```