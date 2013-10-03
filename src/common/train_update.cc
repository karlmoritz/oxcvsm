// File: train_update.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 16-01-2013
// Last Update: Thu 03 Oct 2013 11:45:31 AM BST

// STL
#include <iostream>
#include <random>

#include "shared_defs.h"

// L-BFGS
#include <lbfgs.h>

// Local
#include "train_update.h"
#include "models.h"

lbfgsfloatval_t computeCostAndGrad( Model &model, const lbfgsfloatval_t *x,
    lbfgsfloatval_t *gradient_location, int n)
{
  /******************************************************************************
   *                Check for bimodel setup - otherwise proceed                 *
   ******************************************************************************/

  BackpropagatorBase* lblprop = (model.rae.config.calc_lbl) ? model.rae.getBackpropagator(model,n,1) : nullptr;
  BackpropagatorBase* raeprop = (model.rae.config.calc_rae) ? model.rae.getBackpropagator(model,n,0) : nullptr;

  bool use_full_corpus = false;
  if (model.rae.config.num_sentences == 0) use_full_corpus = true;

#pragma omp parallel for schedule(dynamic)
  for (auto i = model.from; i<model.to; ++i)
  {
    int j = model.indexes[i];

    VectorReal x(model.rae.config.word_representation_size);
    if (model.rae.config.calc_lbl) lblprop->backPropagateLbl(j,x);
    if (model.rae.config.calc_rae) raeprop->backPropagateRae(j,x);
  }
  if (model.rae.config.calc_lbl) lblprop->printInfo();


  if (model.rae.config.calc_lbl) lblprop->normalize(1);
  if (model.rae.config.calc_rae) raeprop->normalize(0);

  lbfgsfloatval_t error_ = (model.calc_L2) ? model.rae.getLambdaCost(model.bools) : 0.0;
  if (model.rae.config.calc_lbl) error_ += lblprop->getError();
  if (model.rae.config.calc_rae) error_ += raeprop->getError();

  int outwidth = 16;
  cout << setw(outwidth) << "ERRORS";
  if (model.rae.config.calc_lbl) cout << setw(outwidth) << "A(lbl)";
  if (model.rae.config.calc_rae) cout << setw(outwidth) << "A(rae)";
  cout << endl;

  cout << setw(outwidth) << " ";
  if (model.rae.config.calc_lbl) cout << setw(outwidth) << lblprop->getError();
  if (model.rae.config.calc_rae) cout << setw(outwidth) << raeprop->getError();
  cout << endl;

  WeightVectorType weights(gradient_location,n);
  weights.setZero();
  if (model.rae.config.calc_lbl) weights += lblprop->dump();
  if (model.rae.config.calc_rae) weights += raeprop->dump();

  delete lblprop;
  delete raeprop;

  if (model.calc_L2)  model.rae.addLambdaGrad(gradient_location,model.bools);

  return error_;
}

void testModel(Model &model)
{

  /***************************************************************************
   *             Define a couple of frequently needed variables              *
   ***************************************************************************/

  int num_sentences = model.rae.config.num_sentences;
  bool use_full_corpus = false;
  if (num_sentences == 0)
  {
    num_sentences = model.corpus.size();
    use_full_corpus = true;
  }
  else
    num_sentences = min(num_sentences,int(model.corpus.size()));

  int correctly_classified_sent = 0;

#pragma omp parallel for schedule(dynamic)
  for (auto i = 0; i<num_sentences; ++i)
  {
    int j = model.indexes[i];

    SinglePropBase* propagator = nullptr;

    if(model.rae.config.tree == TREE_CCG or model.rae.config.tree == TREE_STANFORD)
      propagator = model.rae.getSingleProp(model.corpus[j],0.5,model.bools);

    assert (propagator != nullptr);

    propagator->forwardPropagate(true);
#pragma omp critical
    {
      correctly_classified_sent += propagator->evaluateSentence();
    }
    delete propagator;
  }

  cout << "  " << correctly_classified_sent << "/" << num_sentences << " ";
}

void setVarsAndNumber(Real *&vars, int &number_vars, Model &model)
{
  number_vars += model.rae.theta_size_;
  vars = model.rae.theta_;
}
