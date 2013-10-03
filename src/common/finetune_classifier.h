// File: finetune_classifier.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 29-01-2013
// Last Update: Thu 03 Oct 2013 11:23:44 AM BST

/*
 * Description: Additional logistic regression training on the top part
 * of the model (i.e. after the RAE training).
 * Input is variable (e.g. top-layer plus average of all layers)
 * Output is the number of classes (1 in binary classifier)
 */

#ifndef FINETUNE_CLASSIFIER_H_6NDFQBY1
#define FINETUNE_CLASSIFIER_H_6NDFQBY1

// L-BFGS
#include <lbfgs.h>

// Local
#include "shared_defs.h"
#include "config.h"
//#include "sentsim.h"


class FinetuneClassifier
{
public:
  FinetuneClassifier(RecursiveAutoencoderBase& rae, TrainingCorpus& trainC, float lambdaF,
    float alpha, int dynamic_mode, int iterations);

  ~FinetuneClassifier();

  // Subfunctions for LBFGS and SGD training respectively
  void trainLbfgs(LineSearchType linesearch);
  void trainAdaGrad();

  void evaluate();

protected:
  static int lbfgs_progress_(
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

  static lbfgsfloatval_t lbfgs_evaluate_(
      void *instance,
      const lbfgsfloatval_t *x, // Variables (theta)
      lbfgsfloatval_t *g,       // Put gradient here
      const int n,              // Number of variables
      const lbfgsfloatval_t step);  // line-search step used in this iteration

public:

  lbfgsfloatval_t finetuneCostAndGrad_(
      const lbfgsfloatval_t *x,
      lbfgsfloatval_t *g,
      const int n);


private:


  Real*   trainI_;
  Real*   theta_;

  struct VectorLabelPair {
    WeightVectorType vector;
    Real             label;
    VectorLabelPair(WeightVectorType v, Real l) : vector(v), label(l) {}
  };

  vector<int> mix;

  vector<VectorLabelPair> trainData;

  WeightMatricesType    Wcat;   // Weight matrices for labels
  WeightVectorsType     Bcat;   // Bias weights for labels

  int dynamic_embedding_size;
  int train_length;
  int label_width;
  int num_label_types;
  int theta_size_;

  float lambda;
  float alpha_rae;
  int mode;

  int batch_from;
  int batch_to;

  int iterations;

public:
  int it_count; // counts iterations, restart lbfgs if below
  int num_batches;
  float eta;
  /***************************************************************************
 *                              Serialization                              *
 ***************************************************************************/

  friend class boost::serialization::access;
  template<class Archive>
    void save(Archive& ar, const unsigned version) const {
      ar & theta_size_;
      ar & boost::serialization::make_array(theta_, theta_size_);
    }

  template<class Archive>
    void load(Archive& ar, const unsigned version) {
      delete [] theta_;
      theta_ = new Real[theta_size_];
      ar & boost::serialization::make_array(theta_, theta_size_);
    }
  BOOST_SERIALIZATION_SPLIT_MEMBER()

};

#endif /* end of include guard: FINETUNE_CLASSIFIER_H_6NDFQBY1 */
