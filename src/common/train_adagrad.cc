// File: train_adagrad.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Thu 03 Oct 2013 11:48:28 AM BST

// STL
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>

// Boost
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

// L-BFGS
#include <lbfgs.h>

// Local
#include "train_adagrad.h"
#include "train_update.h"
#include "recursive_autoencoder.h"
#include "utils.h"
#include "fast_math.h"

using namespace std;
namespace bpo = boost::program_options;


int train_adagrad(Model &model, int iterations, float eta, int batches, Real lambda)
{
  Real* vars = nullptr;
  int number_vars = 0;

  setVarsAndNumber(vars,number_vars,model);
  WeightArrayType theta(vars,number_vars);

  Real* Gt_d = new Real[number_vars]();
  // Real* Ginv_d = new Real[number_vars]();
  WeightArrayType Gt(Gt_d,number_vars);
  // WeightArrayType Ginv(Ginv_d,number_vars);
  // Gt.setOnes(); // initialize to ones..

  Real* gradient = new Real[number_vars]();
  lbfgsfloatval_t error = 0;

  // Remove L2 regularization as AdaGrad uses L1 instead
  model.calc_L2 = false;

  int size = model.corpus.size();
  int num_batches = min(batches,size/2);
  int batchsize = max((size/num_batches),2);
  // eta = eta / num_batches;
  cout << "Batch size: " << batchsize << "  eta " << eta << endl;

  Real update;
  Real l1_reg;

  for (auto iteration = 0; iteration < iterations; ++iteration)
  {
    cout << "Iteration " << iteration << endl;
    std::random_shuffle ( model.indexes.begin(), model.indexes.end() );
    for (auto batch = 0; batch < num_batches; ++batch)
    {
      model.from = batch*batchsize;
      model.to = min((batch+1)*batchsize,size);
      error = computeCostAndGrad(model,nullptr,gradient,number_vars);

      // seeing that I need to iterate ...
      for (int i = 0; i < number_vars; ++i) {
        Gt_d[i] += gradient[i]*gradient[i];
        // Update weight: ( eta / \sqrt(Sum square gradients) ) * gradient
        if (Gt_d[i] != 0) {
        update = vars[i] - ((eta / (sqrt(Gt_d[i]))) * gradient[i]);
        l1_reg = (eta / (sqrt(Gt_d[i]))) * lambda;
        vars[i] = signum(update) * max(0.0, abs(update) - l1_reg);
        }
      }
    }

    if (iteration % model.rae.config.dump_freq == 0)
    {
      printf("Dumping model ...\n");
      dumpModel(model,iteration);
    }
  }

  delete [] gradient;
  // delete [] Ginv_d;
  delete [] Gt_d;
  return 0;
}

