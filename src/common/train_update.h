// File: train_update.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 16-01-2013
// Last Update: Thu 03 Oct 2013 11:42:38 AM BST

#ifndef TRAIN_UPDATE_H_VKAFJHPG
#define TRAIN_UPDATE_H_VKAFJHPG

// L-BFGS
#include <lbfgs.h>

// Local
#include "shared_defs.h"

lbfgsfloatval_t computeCostAndGrad(
    Model &model,
    const lbfgsfloatval_t *x, // Variables (theta)
    lbfgsfloatval_t *g,       // Put gradient here
    int n);                   // Number of variables

void testModel(Model &model);

void setVarsAndNumber(Real *&vars, int &number_vars, Model &model);

#endif /* end of include guard: TRAIN_UPDATE_H_VKAFJHPG */
