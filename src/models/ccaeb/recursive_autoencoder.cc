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

namespace ccaeb
{

RecursiveAutoencoder::RecursiveAutoencoder(const ModelData& config, Lambdas lambdas) :
  RecursiveAutoencoderBase(config,lambdas), Theta_Wd(0,0), Theta_Wdr(0,0),
  Theta_Bd(0,0), Theta_Bdr(0,0), Theta_Wl(0,0), Theta_Bl(0,0) {}

RecursiveAutoencoder::RecursiveAutoencoder(const ModelData& config) :
  RecursiveAutoencoderBase(config), Theta_Wd(0,0), Theta_Wdr(0,0),
  Theta_Bd(0,0), Theta_Bdr(0,0), Theta_Wl(0,0), Theta_Bl(0,0) {}

RecursiveAutoencoderBase* RecursiveAutoencoder::cloneEmpty()  {
  return new RecursiveAutoencoder(config,lambdas);
}

BackpropagatorBase* RecursiveAutoencoder::getBackpropagator(Model &model, int n, int type) // doesn't need type. 
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

  int word_width = config.word_representation_size;
  int label_width = config.label_class_size;
  int dict_size = getDictSize();

  theta_D_size_   = word_width * dict_size;
  theta_Wd_size_  = 2 * (num_ccg_rules * word_width * word_width);
  theta_Wdr_size_ = 2 * (num_ccg_rules * word_width * word_width);
  theta_Bd_size_  =     (num_ccg_rules * word_width);
  theta_Bdr_size_ = 2 * (num_ccg_rules * word_width);
  theta_Wl_size_  = word_width * label_width;
  theta_Bl_size_  = label_width;

  theta_size_ = theta_D_size_ + theta_Wd_size_ + theta_Wdr_size_ + theta_Bd_size_ + theta_Bdr_size_ + theta_Wl_size_ + theta_Bl_size_;

  if (create_new_theta)
    theta_ = new Real[theta_size_];
  theta_D_ = theta_;
  theta_Wd_ = theta_D_ + theta_D_size_;
  theta_Wdr_ = theta_Wd_ + theta_Wd_size_;
  theta_Bd_ = theta_Wdr_ + theta_Wdr_size_;
  theta_Bdr_ = theta_Bd_ + theta_Bd_size_;
  theta_Wl_  = theta_Bdr_ + theta_Bdr_size_;
  theta_Bl_  = theta_Wl_ + theta_Wl_size_;
  assert (theta_Bl_ + theta_Bl_size_ == theta_ + theta_size_);

  Real* ptr = theta_;
  new (&Theta_Full) WeightVectorType(ptr, theta_size_);
  
  if (init_words)
    Theta_Full.setZero(); // just be safe (if we add down there instead of initializing)

  new (&Theta_D) WeightVectorType(ptr, theta_D_size_); ptr += theta_D_size_;
  new (&Theta_Wd) WeightVectorType(ptr, theta_Wd_size_); ptr += theta_Wd_size_;
  new (&Theta_Wdr) WeightVectorType(ptr, theta_Wdr_size_); ptr += theta_Wdr_size_;
  new (&Theta_Bd) WeightVectorType(ptr, theta_Bd_size_); ptr += theta_Bd_size_;
  new (&Theta_Bdr) WeightVectorType(ptr, theta_Bdr_size_); ptr += theta_Bdr_size_;
  new (&Theta_Wl) WeightVectorType(ptr, theta_Wl_size_); ptr += theta_Wl_size_;
  new (&Theta_Bl) WeightVectorType(ptr, theta_Bl_size_); ptr += theta_Bl_size_;
  assert (ptr == theta_ + theta_size_);

  //std::random_device rd;
  //std::mt19937 gen(rd());
  std::mt19937 gen(0);
  float r1 = 6.0 / sqrt( 2 * word_width );
  float r2 = 1.0 / sqrt( word_width );
  std::uniform_real_distribution<> dis1(-r1,r1);
  std::uniform_real_distribution<> dis2(-r2,r2);
  std::uniform_real_distribution<> d_sud(0,1);    // Matlab rand / standard uniform distribution
  std::normal_distribution<> d_snd(0,1);          // Matlab randn / standard normal distribution

  ptr = theta_;

  // Initialize Domain
  new (&D) WeightMatrixType(ptr, dict_size, word_width);
  if (init_words) { for (int i=0; i<theta_D_size_; ++i) Theta_D(i) = 0.1 * d_snd(gen); }
  ptr += theta_D_size_;

  // Initialize Domain Matrices
  Wd0.clear();
  Wd1.clear();
  for (auto i=0; i<num_ccg_rules; ++i) {
    Wd0.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;
    Wd1.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;
    if (init_words and config.init_to_I) { 
      Wd0.back().setZero(); Wd0.back().setIdentity(); Wd0.back() *= 0.2;
      Wd1.back().setZero(); Wd1.back().setIdentity(); Wd1.back() *= 0.2;
    }
  }
  if (init_words and not config.init_to_I)
   for (int i=0; i<theta_Wd_size_; ++i) Theta_Wd(i) = dis1(gen);

  // Initialize Domain Reconstruction Matrices
  Wdr0.clear();
  Wdr1.clear();
  for (auto i=0; i<num_ccg_rules; ++i) {
    Wdr0.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;
    Wdr1.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;
    if (init_words and config.init_to_I) { 
      Wdr0.back().setZero(); Wdr0.back().setIdentity(); Wdr0.back() *= 0.2;
      Wdr1.back().setZero(); Wdr1.back().setIdentity(); Wdr1.back() *= 0.2;
    }
  }
  if (init_words and not config.init_to_I)
   for (int i=0; i<theta_Wdr_size_; ++i) Theta_Wdr(i) = dis1(gen);

  // Initialize Domain Biases
  Bd0.clear();
  for (auto i=0; i<num_ccg_rules; ++i) {
    Bd0.push_back(WeightVectorType(ptr, word_width));
    ptr += word_width;
    if (init_words) { Bd0.back().setZero(); }
  }

  // Initialize Domain Reconstruction Biases
  Bdr0.clear();
  Bdr1.clear();
  for (auto i=0; i<num_ccg_rules; ++i) {
    Bdr0.push_back(WeightVectorType(ptr, word_width));
    ptr += word_width;
    Bdr1.push_back(WeightVectorType(ptr, word_width));
    ptr += word_width;
    if (init_words) { 
      Bdr0.back().setZero();
      Bdr1.back().setZero();
    }
  }

  // Initialize Label Matrix and Weight
  Wl.clear();
  Wl.push_back(WeightMatrixType(ptr, label_width, word_width));
  ptr += label_width*word_width;
  if (init_words) { 
    for (int i=0; i<theta_Wl_size_; ++i) Theta_Wl(i) = dis1(gen);
  }

  Bl.clear();
  Bl.push_back(WeightVectorType(ptr, label_width));
  ptr += label_width;
  if (init_words) { 
    Bl.back().setZero();
  }

  assert(ptr == theta_+theta_size_); 


}

RecursiveAutoencoder::~RecursiveAutoencoder() {}

Real RecursiveAutoencoder::getLambdaCost(Bools l)
{
  Real lcost = 0.0;
  if (l.D)   lcost += lambdas.D   * 0.5 * Theta_D.cwiseProduct(Theta_D).sum();
  if (l.Wd)  lcost += lambdas.alpha_rae * lambdas.Wd  * 0.5 * Theta_Wd.cwiseProduct(Theta_Wd).sum();
  if (l.Wdr) lcost += lambdas.alpha_rae * lambdas.Wdr * 0.5 * Theta_Wdr.cwiseProduct(Theta_Wdr).sum();
  if (l.Bd)  lcost += lambdas.alpha_rae * lambdas.Bd  * 0.5 * Theta_Bd.cwiseProduct(Theta_Bd).sum();
  if (l.Bdr) lcost += lambdas.alpha_rae * lambdas.Bdr * 0.5 * Theta_Bdr.cwiseProduct(Theta_Bdr).sum();
  if (l.Wl)  lcost += lambdas.alpha_lbl * lambdas.Wl  * 0.5 * Theta_Wl.cwiseProduct(Theta_Wl).sum();
  if (l.Bl)  lcost += lambdas.alpha_lbl * lambdas.Bl  * 0.5 * Theta_Bl.cwiseProduct(Theta_Bl).sum();
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
  if (l.Wd)
  {
    WeightVectorType X = WeightVectorType(theta_data,theta_Wd_size_); 
    X += (Theta_Wd * lambdas.alpha_rae * lambdas.Wd);
    theta_data += theta_Wd_size_;
  }
  if (l.Wdr)
  {
    WeightVectorType X = WeightVectorType(theta_data,theta_Wdr_size_); 
    X += (Theta_Wdr * lambdas.alpha_rae * lambdas.Wdr);
    theta_data += theta_Wdr_size_;
  }
  if (l.Bd)
  {
    WeightVectorType X = WeightVectorType(theta_data,theta_Bd_size_); 
    X += (Theta_Bd * lambdas.alpha_rae * lambdas.Bd);
    theta_data += theta_Bd_size_;
  }
  if (l.Bdr)
  {
    WeightVectorType X = WeightVectorType(theta_data,theta_Bdr_size_); 
    X += (Theta_Bdr * lambdas.alpha_rae * lambdas.Bdr);
    theta_data += theta_Bdr_size_;
  }
  if (l.Wl)
  {
    WeightVectorType X = WeightVectorType(theta_data,theta_Wl_size_); 
    X += (Theta_Wl * lambdas.alpha_lbl * lambdas.Wl);
    theta_data += theta_Wl_size_;
  }
  if (l.Bl)
  {
    WeightVectorType X = WeightVectorType(theta_data,theta_Bl_size_); 
    X += (Theta_Bl * lambdas.alpha_lbl * lambdas.Bl);
    theta_data += theta_Bl_size_;
  }
}

void RecursiveAutoencoder::setIncrementalCounts(Counts &counts, Real *&vars, int &number)
{
  vars = theta_;
  counts.D   = theta_D_size_;
  counts.Wd  = counts.D   + theta_Wd_size_;
  counts.Wdr = counts.Wd  + theta_Wdr_size_;
  counts.Bd  = counts.Wdr + theta_Bd_size_;
  counts.Bdr = counts.Bd  + theta_Bdr_size_;
  counts.Wl  = counts.Bdr + theta_Wl_size_;
  counts.Bl  = counts.Wl  + theta_Bl_size_;
  number = theta_size_;

}

}

