// File: recursive_autoencoder.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 02-01-2013
// Last Update: Tue 01 Oct 2013 07:05:44 PM BST

/*------------------------------------------------------------------------
 * Description: <DESC> 
 * 
 *------------------------------------------------------------------------
 * History:
 * TODO:    
 *========================================================================
 */

// Local
#include "recursive_autoencoder.h"
#include "backpropagator.h"

#include <iostream>

// Namespaces
using namespace std;

namespace mvrnn
{

RecursiveAutoencoder::RecursiveAutoencoder(const ModelData& config, Lambdas
    lambdas) : RecursiveAutoencoderBase(config,lambdas), W(0,0,0), A(0,0),
  Theta_U(0,0), Theta_V(0,0), Theta_W(0,0), Theta_Wd(0,0), Theta_Wf(0,0),
  Theta_Wl(0,0) {}

RecursiveAutoencoder::RecursiveAutoencoder(const ModelData& config) :
  RecursiveAutoencoderBase(config), W(0,0,0), A(0,0), Theta_U(0,0),
  Theta_V(0,0), Theta_W(0,0), Theta_Wd(0,0), Theta_Wf(0,0), Theta_Wl(0,0) {}

RecursiveAutoencoderBase* RecursiveAutoencoder::cloneEmpty()  {
  return new RecursiveAutoencoder(config,lambdas);
}

BackpropagatorBase* RecursiveAutoencoder::getBackpropagator(Model &model, int n, int type)
{
  BackpropagatorBase* bpb = new Backpropagator(this, model, n);
  return bpb;
}

SinglePropBase* RecursiveAutoencoder::getSingleProp(const TrainingInstance &t, Real beta, Bools updates)
{
  SinglePropBase* spb = new SingleProp(this, t, beta, updates);
  return spb;
}

void RecursiveAutoencoder::init(bool init_words, bool create_new_theta) {

  int num_weight_types_ = 1;

  int word_width = config.word_representation_size;
  int label_width = config.label_class_size;
  int dict_size = getDictSize();
  int approx_width = config.approx_width;

  theta_D_size_ = word_width * dict_size;
  theta_U_size_ = word_width * approx_width * dict_size;
  theta_V_size_ = word_width * approx_width * dict_size;
  theta_W_size_ = word_width * dict_size;
  theta_A_size_ = dict_size;
  theta_L_size_ = theta_D_size_ + theta_U_size_ + theta_V_size_ + theta_W_size_ + theta_A_size_;

  theta_Wd_size_ = num_weight_types_ * 2*word_width * word_width;
  theta_Wf_size_ = num_weight_types_ * 2*word_width * word_width;
  theta_Wl_size_ = word_width * label_width;

  theta_size_ = theta_L_size_ + theta_Wd_size_ + theta_Wf_size_ + theta_Wl_size_;

  // Add some asserts here perhaps!

  if (create_new_theta)
    theta_ = new Real[theta_size_];
  theta_D_ = theta_;
  theta_U_ = theta_D_ + theta_D_size_;
  theta_V_ = theta_U_ + theta_U_size_;
  theta_W_ = theta_V_ + theta_V_size_;
  theta_A_ = theta_W_ + theta_W_size_;
  theta_Wd_ = theta_A_ + theta_A_size_; 
  theta_Wf_ = theta_Wd_ + theta_Wd_size_;
  theta_Wl_ = theta_Wf_ + theta_Wf_size_;

  Real* ptr = theta_;
  new (&Theta_Full) WeightVectorType(ptr, theta_size_);

  if (init_words)
    Theta_Full.setZero(); // just be safe (if we add down there instead of initializing)

  new (&Theta_D) WeightVectorType(ptr, theta_D_size_); ptr += theta_D_size_;
  new (&Theta_U) WeightVectorType(ptr, theta_U_size_); ptr += theta_U_size_;
  new (&Theta_V) WeightVectorType(ptr, theta_V_size_); ptr += theta_V_size_;
  new (&Theta_W) WeightVectorType(ptr, theta_W_size_); ptr += theta_W_size_;
  new (&A) WeightVectorType(ptr, theta_A_size_); ptr += theta_A_size_;
  new (&Theta_Wd) WeightVectorType(ptr, theta_Wd_size_); ptr += theta_Wd_size_;
  new (&Theta_Wf) WeightVectorType(ptr, theta_Wf_size_); ptr += theta_Wf_size_;
  new (&Theta_Wl) WeightVectorType(ptr, theta_Wl_size_); ptr += theta_Wl_size_;

  std::random_device rd;
  std::mt19937 gen(rd());
  //std::mt19937 gen(0);
  float r1 = 1.0 / sqrt( 2 * word_width );
  float r2 = 1.0 / sqrt( word_width );
  std::uniform_real_distribution<> dis1(-r1,r1);
  std::uniform_real_distribution<> dis2(-r2,r2);
  std::uniform_real_distribution<> d_sud(0,1);    // Matlab rand / standard uniform distribution
  std::normal_distribution<> d_snd(0,1);          // Matlab randn / standard normal distribution

  ptr = theta_;

  // Clear all matrices
  U.clear();
  V.clear();
  Wd0.clear();
  Wd1.clear();
  Wf0.clear();
  Wf1.clear();
  Wl.clear();

  // Initialize Domain
  new (&D) WeightMatrixType(ptr, dict_size, word_width);
  if (init_words) { for (int i=0; i<theta_D_size_; ++i) Theta_D(i) = 0.1 * d_snd(gen); }
  ptr += theta_D_size_;

  // Initialize Function
  for (auto i=0; i<dict_size; ++i) {
    U.push_back(WeightMatrixType(ptr, word_width, approx_width));
    ptr += word_width*approx_width;
  }
  if (init_words) { for (int i=0; i<theta_U_size_; ++i) Theta_U(i) = 0.01 * d_snd(gen); }

  for (auto i=0; i<dict_size; ++i) {
    V.push_back(WeightMatrixType(ptr, approx_width, word_width));
    ptr += word_width*approx_width;
  }
  if (init_words) { for (int i=0; i<theta_V_size_; ++i) Theta_V(i) = 0.01 * d_snd(gen); }

  // Initialize Function Diagonal
  new (&W) WeightMatrixType(ptr, dict_size, word_width);
  if (init_words) { for (int i=0; i<theta_W_size_; ++i) Theta_W(i) = 1; } // initialize diagonal to one 
  ptr += theta_W_size_;

  // Initialize Function-Composition Weight
  //A = Theta_A;
  ptr += theta_A_size_;
  if (init_words) { for (int i=0; i<theta_A_size_; ++i) A(i) = 0; }

  // Initialize Domain-Composition matrices
  for (auto i=0; i<num_weight_types_; ++i) {
    Wd0.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;
    Wd1.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;
    if (init_words) {
      Wd0.back().setZero(); Wd0.back().setIdentity(); Wd0.back() *= 0.2;
      Wd1.back().setZero(); Wd1.back().setIdentity(); Wd1.back() *= 0.2;
    }
  }
  //for (int i=0; i<theta_Wd_size_; ++i) Theta_Wd(i) = dis(gen);

  // Initialize Function-Composition matrices
  for (auto i=0; i<num_weight_types_; ++i) {
    Wf0.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;
    Wf1.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;
  }
  if (init_words) { for (int i=0; i<theta_Wf_size_; ++i) Theta_Wf(i) = dis2(gen); }

  // Initialize Label matrices
  for (auto i=0; i<1; ++i) {
    Wl.push_back(WeightMatrixType(ptr, label_width, word_width));
    ptr += word_width*label_width;
  }
  if (init_words) { for (int i=0; i<theta_Wl_size_; ++i) Theta_Wl(i) = dis1(gen); }

  assert(ptr == theta_+theta_size_); 

}

RecursiveAutoencoder::~RecursiveAutoencoder() {}

Real RecursiveAutoencoder::getLambdaCost(Bools l)
{
  Real lcost = 0.0;
  if (l.D)  lcost += lambdas.D * 0.5 * Theta_D.cwiseProduct(Theta_D).sum();
  if (l.U)  lcost += lambdas.U * 0.5 * Theta_U.cwiseProduct(Theta_U).sum();
  if (l.V)  lcost += lambdas.V * 0.5 * Theta_V.cwiseProduct(Theta_V).sum();
  if (l.W)  lcost += lambdas.W * 0.5 * Theta_W.cwiseProduct(Theta_W).sum();
  if (l.Wd)  lcost += lambdas.Wd * 0.5 * Theta_Wd.cwiseProduct(Theta_Wd).sum();
  if (l.Wf)  lcost += lambdas.Wf * 0.5 * Theta_Wf.cwiseProduct(Theta_Wf).sum();
  if (l.Wl)  lcost += lambdas.Wl * 0.5 * Theta_Wl.cwiseProduct(Theta_Wl).sum();
  return lcost;
}

void RecursiveAutoencoder::addLambdaGrad(Real* theta_data, Bools l)
{
  if (l.D)
  {
    WeightVectorType X = WeightVectorType(theta_data,theta_D_size_); 
    X += (Theta_D * lambdas.D);
    theta_data += theta_D_size_;
  }
  if (l.U)
  {
    WeightVectorType X = WeightVectorType(theta_data,theta_U_size_); 
    X += (Theta_U * lambdas.U);
    theta_data += theta_U_size_;
  }
  if (l.V)
  {
    WeightVectorType X = WeightVectorType(theta_data,theta_V_size_); 
    X += (Theta_V * lambdas.V);
    theta_data += theta_V_size_;
  }
  if (l.W)
  {
    WeightVectorType X = WeightVectorType(theta_data,theta_W_size_); 
    X += (Theta_W * lambdas.W);
    theta_data += theta_W_size_;
  }
  if (l.A)
  {
    //WeightVectorType X = WeightVectorType(theta_data,theta_A_size_); 
    //X += (A * lambdas.A);
    theta_data += theta_A_size_;
  }
  if (l.Wd)
  {
    WeightVectorType X = WeightVectorType(theta_data,theta_Wd_size_); 
    X += (Theta_Wd * lambdas.Wd);
    theta_data += theta_Wd_size_;
  }
  if (l.Wf)
  {
    WeightVectorType X = WeightVectorType(theta_data,theta_Wf_size_); 
    X += (Theta_Wf * lambdas.Wf);
    theta_data += theta_Wf_size_;
  }
  if (l.Wl)
  {
    WeightVectorType X = WeightVectorType(theta_data,theta_Wl_size_); 
    X += (Theta_Wl * lambdas.Wl);
    theta_data += theta_Wl_size_;
  }
}


void RecursiveAutoencoder::setIncrementalCounts(Counts &counts, Real *&vars, int &number)
{
  vars = theta_;
  counts.D   = theta_D_size_;
  counts.U   = counts.D   + theta_U_size_;
  counts.V   = counts.U   + theta_V_size_;
  counts.W   = counts.V   + theta_W_size_;
  counts.A   = counts.W   + theta_A_size_;
  counts.Wd  = counts.A   + theta_Wd_size_;
  counts.Wf  = counts.Wd  + theta_Wf_size_;
  counts.Wl  = counts.Wf  + theta_Wl_size_;
  number = theta_size_;

}

}
