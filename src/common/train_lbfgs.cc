// File: train_lbfgs.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 09-01-2013
// Last Update: Thu 03 Oct 2013 11:48:59 AM BST

// STL
#include <iostream>

// Boost
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include "shared_defs.h"

// L-BFGS
#include <lbfgs.h>

// Local
#include "train_lbfgs.h"
#include "train_update.h"

#include "utils.h"
#include "recursive_autoencoder.h"
#include "finite_grad_check.h"

using namespace std;
namespace bpo = boost::program_options;


int train_lbfgs(Model& model, LineSearchType linesearch, int max_iterations, float epsilon)
{
  /***************************************************************************
   *                              BFGS training                              *
   ***************************************************************************/

  lbfgs_parameter_t param;
  lbfgs_parameter_init(&param);
  param.linesearch = linesearch;
  param.max_iterations = max_iterations;
  param.epsilon = epsilon;
  param.m = 25;

  Real* vars = nullptr;
  int number_vars = 0;

  setVarsAndNumber(vars,number_vars,model);

  const int n = number_vars;
  Real error = 0.0;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> rand(1, model.to-1);

  model.noise_sample_offset = rand(rd);

  while (model.it_count < max_iterations) {
    param.max_iterations = max_iterations - model.it_count;
    cout << "Starting new L-BFGS optimization" << endl;
    int ret = lbfgs(n, vars, &error, evaluate_, progress, &model, &param);
    cout << "L-BFGS optimization terminated with status code = " << ret << endl;
    cout << "fx=" << error << endl;
  }

  return 0;
}

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
    )
{
  printf("Iteration %d:\n", k);
  printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
  printf("\n");

  Model* model = reinterpret_cast<Model*>(instance);
  ++model->it_count;
  if (k % model->rae.config.dump_freq == 0)
  {
    printf("Dumping model ...\n");
    dumpModel(*model,k);
  }

  return 0;
}

lbfgsfloatval_t evaluate_(
    void *instance,
    const lbfgsfloatval_t *x,   // Variables (theta)
    lbfgsfloatval_t *g,         // Put gradient here
    const int n,                // Number of variables
    const lbfgsfloatval_t step) // line-search step used in this iteration
{
  Model* model = reinterpret_cast<Model*>(instance);
  //finite_grad_check(*model);
  return computeCostAndGrad(*model,x,g,n);
}


int train_lbfgs_minibatch(Model& model, LineSearchType linesearch, int
    max_iterations, float epsilon, int batches)
{
  /***************************************************************************
   *                              BFGS training                              *
   ***************************************************************************/

  lbfgs_parameter_t param;
  lbfgs_parameter_init(&param);
  param.linesearch = linesearch;
  param.max_iterations = 5;
  param.epsilon = epsilon;
  param.m = 5;

  int size = model.corpus.size();
  int num_batches = min(batches,size);
  int batchsize = (size / num_batches) + 1;

  cout << "Batch size: " << batchsize << " on " << num_batches << " batches." << endl;
  Real* vars = nullptr;

  int number_vars = 0;
  setVarsAndNumber(vars,number_vars,model);
  const int n = number_vars;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> rand(1, model.to-1);

  Real error = 0.0;

  for (auto iteration = 0; iteration < max_iterations; ++iteration)
  {
    cout << "Iteration " << iteration << endl;
    std::random_shuffle ( model.indexes.begin(), model.indexes.end() );
    for (auto batch = 0; batch < num_batches; ++batch)
    {
      cout << "Starting next batch" << endl;
      model.from = batch*batchsize;
      model.to = min((batch+1)*batchsize,size);
      model.noise_sample_offset = rand(rd);
      int ret = lbfgs(n, vars, &error, evaluate_, progress_minibatch, &model, &param);
      cout << "L-BFGSi minibatch optimization terminated with status code = " << ret << endl;
      cout << "fx=" << error << endl;
    }
    if (iteration % model.rae.config.dump_freq == 0)
    {
      printf("Dumping model ...\n");
      dumpModel(model,iteration);
    }
  }
  return 0;
}

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
    )
{
  printf("Iteration %d:\n", k);
  printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
  printf("\n");

  Model* model = reinterpret_cast<Model*>(instance);
  ++model->it_count;

  return 0;
}
