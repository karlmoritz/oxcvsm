// File: train_adagrad.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 09-01-2013
// Last Update: Thu 03 Oct 2013 11:24:00 AM BST

#ifndef TRAIN_ADAGRAD_H_1TFAVYZ6
#define TRAIN_ADAGRAD_H_1TFAVYZ6

// Local
#include "shared_defs.h"

int train_adagrad(Model &model, int iterations, float eta, int batches, Real lambda=0.0001);

#endif /* end of include guard: TRAIN_ADAGRAD_H_1TFAVYZ6 */

