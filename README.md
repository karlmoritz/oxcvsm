oxcvsm
======

Oxford Compositional Vector Space Models

Author: Karl Moritz Hermann

# DEPRECATION NOTE

Please note that there is an updated version of this code on Github:
https://github.com/karlmoritz/bicvm .  That codebase does not contain all of the
models from the 2013 ACL paper, but it is fairly simple to rebuild those as well
as a myriad of others. If you want to extend any of this work, I would
appreciate if you directed your efforts to that version of the code.

## Dependencies

PugiXML (a recent version is included under src/pugi)
Boost libraries
Eigen library

## Getting started

1. Install dependencies
2. Adjust paths in Makefile
3. Compile with 'make'
4. ./train --help

Run a simple test:

./train --input-pos data/demo.pos \
        --input-neg data/demo.neg \
        --type ccaeb --tree ccg \
        --word-width 5 --method fgc

This test should load the two included demo files (positive and negative
sentiment), load these using the CCG format parser, load the CCAEB model and
perform a finite gradient check using embeddings of width 5.

The finite gradient checker currently exits the program on a failed assertion,
hence the final few lines of the output should look something like this:

Wl  2359: 0.0438076 vs 0.0438076   9.99583 - 9.99583[0.636769]
  6/10 Z: 61: (36 / 25)O: 40: (19 / 21)
          ERRORS          A(lbl)          A(rae)
                        0.972626         1.02926
Bl  2360: 0.191106 vs 0.191106   9.99583 - 9.99583[0]
total: 3.21547e-05 D/U/V/W/A/Wd/Wdr/Bd/Bdr/Wf/Wl/Bl
7.69128e-06 0 0 0 0 1.20855e-05 1.07516e-05 7.50228e-07 8.06919e-07 0 5.44982e-08 1.46649e-08
train: src/common/finite_grad_check.cc:85: int finite_grad_check(Model&): Assertion `false' failed.
Aborted (core dumped)

## Additional Information

I will continue updating this package over the coming months. The code in here
differs somewhat from the code used for the experiments in the ACL '13 paper as
I have since re-written a large part of the underlying structure.

At this point, only two models are included in this code, the CCAE-B model from
my paper, as well as a re-implementation of the MV-RNN paper in Socher et al.
2012.

Please use the features provided by Github for any questions and comments, so
that any issues and solutions can be seen by everyone else as well.

## Papers / Citation

If you use this software package in your experiments and publish related work,
please cite the following paper:

@InProceedings{hermann-blunsom:2013:ACL2013,
  author    = {Hermann, Karl Moritz  and  Blunsom, Phil},
  title     = {The Role of Syntax in Vector Space Models of Compositional Semantics},
  booktitle = {Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {August},
  year      = {2013},
  address   = {Sofia, Bulgaria},
  publisher = {Association for Computational Linguistics},
  pages     = {894--904},
  url       = {http://www.aclweb.org/anthology/P13-1088}
}

