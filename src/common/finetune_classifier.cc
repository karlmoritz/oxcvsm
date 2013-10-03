// File: finetune_classifier.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 29-01-2013
// Last Update: Thu 03 Oct 2013 11:26:25 AM BST

// STL
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>

// Boost
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

// L-BFGS
#include <lbfgs.h>

// Local
#include "finetune_classifier.h"
#include "models.h"
#include "fast_math.h"

using namespace std;
namespace bpo = boost::program_options;

FinetuneClassifier::FinetuneClassifier(RecursiveAutoencoderBase& rae,
    TrainingCorpus& trainC, float lambdaF, float alpha,
    int dynamic_mode, int iterations) : lambda(lambdaF), alpha_rae(alpha),
  mode(dynamic_mode), iterations(iterations), it_count(0), num_batches(100),
  eta(0.1) {

  /***************************************************************************
   *             Define a couple of frequently needed variables              *
   ***************************************************************************/

  train_length = min(rae.config.num_sentences,int(trainC.size()));
  bool use_full_corpus = false;
  if (train_length == 0)
  {
    train_length = trainC.size();
    use_full_corpus = true;
  }

  label_width = rae.config.label_class_size;
  num_label_types = rae.config.num_label_types; // 1

  int multiplier = 1;
  if (mode == 0)
    multiplier = 2;
  if (mode == 3)
    multiplier = 3; // only works for compound test
  if (mode == 4)
    multiplier = 4;
  // Embedding: s1 + s2 + cos_sim(s1,s2) + len(s1) + len(s2) +
  // unigram_overlap(s1,s2) following Blacoe/Lapata 2012
  dynamic_embedding_size = multiplier * rae.config.word_representation_size;

  theta_size_ = dynamic_embedding_size * label_width * num_label_types + label_width * num_label_types;

  trainI_ = new Real[train_length * dynamic_embedding_size]();
  theta_  = new Real[theta_size_];
  WeightVectorType theta(theta_,theta_size_);
  theta.setZero();
  if (true) {
    std::random_device rd;
    std::mt19937 gen(rd());
    //std::mt19937 gen(0);
    float r = sqrt( 6.0 / dynamic_embedding_size);
    std::uniform_real_distribution<> dis(-r,r);
    for (int i=0; i<theta_size_; i++)
      theta(i) = dis(gen);
  }

  Real* ptr = trainI_;
  for (auto i=0; i<train_length; ++i) {
    int j = i;
    if ((not use_full_corpus) and (i%2 == 1))
      j = trainC.size() - i;
    trainData.push_back(VectorLabelPair(WeightVectorType(ptr, dynamic_embedding_size),trainC[j].value));
    mix.push_back(i);
    ptr += dynamic_embedding_size;
  }

  ptr = theta_;
  for (auto i=0; i<num_label_types; ++i) {
    Wcat.push_back(WeightMatrixType(ptr, label_width, dynamic_embedding_size));
    ptr += label_width * dynamic_embedding_size;
  }
  for (auto i=0; i<num_label_types; ++i) {
    Bcat.push_back(WeightVectorType(ptr, label_width));
    Bcat.back().setZero(); // discuss ..
    ptr += label_width;
  }

  /***************************************************************************
   *        Populate train input with forward Propagation and tricks         *
   ***************************************************************************/

#pragma omp parallel for schedule(dynamic)
  for (auto i = 0; i<train_length; ++i)
  {
    int j = i;
    if ((not use_full_corpus) and (i%2 == 1))
      j = trainC.size() - i;

    SinglePropBase* propagator = nullptr;

    Bools bools;
    if(rae.config.tree == TREE_CCG or rae.config.tree == TREE_STANFORD)
      propagator = rae.getSingleProp(trainC[j],0.5,bools);
    assert (propagator != nullptr);

    propagator->forwardPropagate(true);
    propagator->setDynamic(trainData[i].vector,mode);
    //cout << "C: " << trainData[i].vector[0] << endl;

    delete propagator;
  }
}

void FinetuneClassifier::evaluate()
{

  vector<VectorLabelPair>& data = trainData;
  int length = train_length;

  int right = 0;
  int wrong = 0;
  int tp = 0;
  int fp = 0;
  int tn = 0;
  int fn = 0;

#pragma omp parallel for schedule(dynamic)
  for (auto i = 0; i<length; ++i)
  {

    // Encode input
    ArrayReal label_vec = (
        Wcat[0] * data[i].vector + Bcat[0]
        ).unaryExpr(std::ptr_fun(getSigmoid)).array();

    ArrayReal lbl_sm = data[i].label - label_vec;

#pragma omp critical
    {
      if(abs(lbl_sm.sum()) > 0.5)
      {
        wrong += 1;
        if(data[i].label == 0)  ++fp;
        else                    ++fn;
      }
      else
      {
        right += 1;
        if(data[i].label == 0)  ++tn;
        else                    ++tp;
      }
    }
  }
    cout << right << "/" << right + wrong << "  ";
    Real precision = 1.0 * tp / (tp + fp);
    Real recall    = 1.0 * tp / (tp + fn);
    Real accuracy  = 1.0 * (tp + tn) / (tp + tn + fp + fn);
    Real f1score   = 2.0 * (precision * recall) / (precision + recall);
    cout << "Acc/F1: " << accuracy << " " << f1score << endl;
}

void FinetuneClassifier::trainLbfgs(LineSearchType linesearch)
{
  batch_from = 0;
  batch_to = train_length;

  lbfgs_parameter_t param;
  lbfgs_parameter_init(&param);
  param.linesearch = linesearch;
  param.max_iterations = iterations;
  //param.epsilon = 0.00000001;
  param.m = 25;

  const int n = theta_size_;
  auto vars = theta_;
  Real error = 0.0;

  int tries = 0;

  while (tries < 3 and it_count < 250)
  {
    int ret = lbfgs(n, vars, &error, lbfgs_evaluate_, lbfgs_progress_, this, &param);
    cout << "L-BFGS optimization terminated with status code = " << ret << endl;
    cout << "fx=" << error << endl;
    ++tries;
  }
}

void FinetuneClassifier::trainAdaGrad()
{
  auto vars = theta_;
  int number_vars = theta_size_;

  WeightArrayType theta(vars,number_vars);

  Real* Gt_d = new Real[number_vars];
  Real* Ginv_d = new Real[number_vars];
  WeightArrayType Gt(Gt_d,number_vars);
  WeightArrayType Ginv(Ginv_d,number_vars);
  Gt.setZero();

  Real* data1 = new Real[number_vars];

  int batchsize = (train_length / num_batches) + 1;
  cout << "Batch size: " << batchsize << endl;

  for (auto iteration = 0; iteration < iterations; ++iteration)
  {
    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    random_shuffle (mix.begin(), mix.end()); //, std::default_random_engine(seed));

    for (auto batch = 0; batch < num_batches; ++batch)
    {
      batch_from = batch*batchsize;
      batch_to = min((batch+1)*batchsize,train_length);

      if (batch_to - batch_from > 0)
      {
        WeightArrayType grad(data1,number_vars);

        //float err =
        finetuneCostAndGrad_(theta_,data1,number_vars);
        //grad /= (batch_to - batch_from);
        Gt += grad*grad;
        for (int i=0;i<number_vars;i++) {
          if (Gt_d[i] == 0)
            Ginv_d[i] = 0;
          else
            Ginv_d[i] = sqrt(1/Gt_d[i]);
        }

        grad *= Ginv;
        grad *= eta;

        //cout << theta.abs().sum() << " vs " << grad.abs().sum() << endl;
        theta -= grad;
      }
    }

    evaluate();
  }

  delete [] data1;
  delete [] Gt_d;
  delete [] Ginv_d;

}

lbfgsfloatval_t FinetuneClassifier::lbfgs_evaluate_(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
  return reinterpret_cast<FinetuneClassifier*>(instance)->finetuneCostAndGrad_(x, g, n);
}

int FinetuneClassifier::lbfgs_progress_(
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
  cout << "N: " << n << endl;
  printf("Iteration %d:\n", k);
  printf("  fx = %f, x[0] = %f, x[1] = %f %f\n", fx, x[0], x[1], x[2]);
  printf("  fx = %f, g[0] = %f, g[1] = %f %f\n", fx, g[0], g[1], g[2]);
  printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
  printf("\n");


  reinterpret_cast<FinetuneClassifier*>(instance)->it_count++;
  reinterpret_cast<FinetuneClassifier*>(instance)->evaluate();
  return 0;
}

// ForwardPropagates and returns error and gradient on self
lbfgsfloatval_t FinetuneClassifier::finetuneCostAndGrad_(
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *gradient_location,
    const int n)
{

  WeightVectorType  grad(gradient_location,theta_size_);
  grad.setZero();

  WeightMatricesType Wcatgrad;
  WeightVectorsType  Bcatgrad;

  Real* ptr = gradient_location;
  for (auto i=0; i<num_label_types; ++i) {
    Wcatgrad.push_back(WeightMatrixType(ptr, label_width, dynamic_embedding_size));
    ptr += label_width * dynamic_embedding_size;
  }
  for (auto i=0; i<num_label_types; ++i) {
    Bcatgrad.push_back(WeightVectorType(ptr, label_width));
    ptr += label_width;
  }

  assert(ptr == theta_size_ + gradient_location);

  /***************************************************************************
   *        Populate train input with forward Propagation and tricks         *
   ***************************************************************************/
  Real cost = 0.0;
  int right = 0;
  int wrong = 0;
  int tp = 0;
  int fp = 0;
  int tn = 0;
  int fn = 0;

#pragma omp parallel for schedule(dynamic)
  for (auto k = batch_from; k<batch_to; ++k)
  {
    auto i = mix[k];

    // Encode input
    ArrayReal label_vec = (
        Wcat[0] * trainData[i].vector + Bcat[0]
        ).unaryExpr(std::ptr_fun(getSigmoid)).array();

    ArrayReal lbl_sm = label_vec - trainData[i].label;
    ArrayReal delta = lbl_sm * (label_vec) * (1 - label_vec);

#pragma omp critical
    {
      //cout << "D/D2" << delta << " " << delta2 << endl;
      cost += 0.5 * (lbl_sm * lbl_sm).sum();
      Wcatgrad[0] += delta.matrix() * trainData[i].vector.transpose();
      Bcatgrad[0] += delta.matrix();

      if(abs(lbl_sm.sum()) > 0.5)
      {
        wrong += 1;
        if(trainData[i].label == 0)
          ++fp;
        else
          ++fn;
      }
      else
      {
        right += 1;
        if(trainData[i].label == 0)
          ++tn;
        else
          ++tp;
      }
    }
  }

  /*
   *cout << "Correct: " << right << "/" << right + wrong << "   ";
   *cout << "Zero: " << tp << "/" << tp+fn << " ";
   *cout << "One:  " << tn << "/" << tn+fp << endl;
   */

  float lambda_partial = lambda * (batch_to - batch_from) / train_length;
  Wcatgrad[0] += lambda_partial*Wcat[0];
  cost += 0.5*lambda_partial*(Wcat[0].cwiseProduct(Wcat[0])).sum();

  Wcatgrad[0] /= (batch_to - batch_from);
  cost /= (batch_to - batch_from);

  return cost;
}


FinetuneClassifier::~FinetuneClassifier()
{
  delete [] trainI_;
  delete [] theta_;

}
