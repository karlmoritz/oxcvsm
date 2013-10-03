// File: backpropagator.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-04-2013
// Last Update: Thu 03 Oct 2013 11:42:48 AM BST

#ifndef BACKPROPAGATOR_H_BQGPXILL
#define BACKPROPAGATOR_H_BQGPXILL

// L-BFGS
#include <lbfgs.h>

// Local
#include "../../common/shared_defs.h"
#include "../../common/backpropagatorbase.h"
#include "recursive_autoencoder.h"
#include "singleprop.h"

namespace mvrnn
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
  void setGradients_(SingleProp* propagator, int i);

  RecursiveAutoencoder* rae_;
  Model& model;
  WeightMatrixType grad_D;
  WeightMatricesType grad_U;
  WeightMatricesType grad_V;
  WeightMatrixType grad_W;
  WeightVectorType grad_A;
  WeightVectorType grad_Wd;
  WeightVectorType grad_Wf;
  WeightVectorType grad_Wl;

  WeightVectorType weights;

  Real* data;

  int word_width;
  int approx_width;
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

/*
 *class BackpropagatorShell : public BackpropagatorBase
 *{
 *public:
 *  BackpropagatorShell ();
 *  ~BackpropagatorShell ();
 *  int  backPropagateLbl(int i, VectorReal& x);
 *  void backPropagateRae(int i, VectorReal& x);
 *  void backPropagateBi(int i,  VectorReal& x, VectorReal word);
 *  void normalize(int type); // type: 0=rae, 1=lbl, 2=bi
 *  void printInfo();
 *  lbfgsfloatval_t getError();
 *  WeightVectorType dump();
 *};
 */

}
#endif /* end of include guard: BACKPROPAGATOR_H_BQGPXILL */
