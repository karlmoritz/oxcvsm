// File: backpropagatorbase.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-04-2013
// Last Update: Thu 03 Oct 2013 11:37:10 AM BST

/*------------------------------------------------------------------------
 * Description: <DESC>
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================
 */

#ifndef BACKPROPAGATORBASE_H_BR0USYKJ
#define BACKPROPAGATORBASE_H_BR0USYKJ


// L-BFGS
#include <lbfgs.h>

// Local
#include "shared_defs.h"
#include "recursive_autoencoder.h"

class BackpropagatorBase
{
public:
  //BackpropagatorBase (RecursiveAutoencoder* rae, Model &model, int n);
  virtual ~BackpropagatorBase () {};

  virtual int  backPropagateLbl(int i, VectorReal& x) = 0;
  virtual void backPropagateRae(int i, VectorReal& x) = 0;

  virtual void normalize(int type) = 0; // type: 0=rae, 1=lbl, 2=bi

  virtual void printInfo() = 0;
  virtual lbfgsfloatval_t getError() = 0;
  virtual void addError(lbfgsfloatval_t i);
  virtual WeightVectorType dump() = 0;

  virtual SinglePropBase* forwardPropagate(int i, VectorReal& x); // forward propagates (on B)

};

#endif /* end of include guard: BACKPROPAGATORBASE_H_BR0USYKJ */
