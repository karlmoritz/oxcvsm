// File: train_lbfgs.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 09-01-2013
// Last Update: Tue 01 Oct 2013 07:09:25 PM BST

/*------------------------------------------------------------------------
 * Description: <DESC> 
 * 
 *------------------------------------------------------------------------
 * History:
 * TODO:    
 *========================================================================
 */

#ifndef TRAIN_LBFGS_H_NPFCBDNG
#define TRAIN_LBFGS_H_NPFCBDNG

// L-BFGS
#include <lbfgs.h>

// Local
#include "shared_defs.h"

static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    );

static int progress_minibatch(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    );


lbfgsfloatval_t evaluate_(
    void *instance,
    const lbfgsfloatval_t *x, // Variables (theta)
    lbfgsfloatval_t *g,       // Put gradient here
    const int n,              // Number of variables
    const lbfgsfloatval_t step);  // line-search step used in this iteration

int train_lbfgs(Model& model, LineSearchType linesearch, int max_iterations, float epsilon);

int train_lbfgs_minibatch(Model& model, LineSearchType linesearch, int
    max_iterations, float epsilon, int batches);

#endif /* end of include guard: TRAIN_LBFGS_H_NPFCBDNG */
