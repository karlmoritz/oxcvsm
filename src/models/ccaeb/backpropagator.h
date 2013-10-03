// File: backpropagator.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-04-2013
// Last Update: Thu 03 Oct 2013 11:39:04 AM BST

/*------------------------------------------------------------------------
 * Description: <DESC>
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================
 */

#ifndef BACKPROPAGATOR_H_ZFKO1GRJ
#define BACKPROPAGATOR_H_ZFKO1GRJ

// L-BFGS
#include <lbfgs.h>

// Local
#include "../../common/shared_defs.h"
#include "../../common/backpropagatorbase.h"
#include "recursive_autoencoder.h"
#include "singleprop.h"

namespace ccaeb
{

class Backpropagator : public BackpropagatorBase
{
public:
  Backpropagator (RecursiveAutoencoder* rae, Model &model, int n);
  ~Backpropagator ();

  int  backPropagateLbl(int i, VectorReal& x);
  void backPropagateRae(int i, VectorReal& x);

  void normalize(int type); // type: 0=rae, 1=lbl

  void printInfo();
  lbfgsfloatval_t getError();
  WeightVectorType dump();

private:

  RecursiveAutoencoder* rae_;
  Model& model;
  WeightMatrixType grad_D;
  WeightVectorType grad_Wd;
  WeightVectorType grad_Wdr;
  WeightVectorType grad_Bd;
  WeightVectorType grad_Bdr;
  WeightVectorType grad_Wl;
  WeightVectorType grad_Bl;

  WeightVectorType weights;

  Real* data;

  int word_width;
  int dict_size;

  int correctly_classified_sent;
  int zero_should_be_one;
  int zero_should_be_zero;
  int one_should_be_zero;
  int one_should_be_one;
  int is_a_zero;
  int is_a_one;

  Real error_;
  int count_nodes_;
  int count_words_;


};

}
#endif /* end of include guard: BACKPROPAGATOR_H_ZFKO1GRJ */
