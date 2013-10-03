// File: backpropagator.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-04-2013
// Last Update: Thu 03 Oct 2013 11:42:52 AM BST

#include "backpropagator.h"

namespace mvrnn
{

Backpropagator::Backpropagator (RecursiveAutoencoder* rae, Model &model,int n) :
  rae_(rae), model(model), grad_D(0,0,0), grad_W(0,0,0), grad_A(0,0),
  grad_Wd(0,0), grad_Wf(0,0), grad_Wl(0,0), weights(0,0),
  correctly_classified_sent(0), zero_should_be_one(0), zero_should_be_zero(0),
  one_should_be_zero(0), one_should_be_one(0), is_a_zero(0), is_a_one(0),
  error_(0.0), count_nodes_(0), count_words_(0)
  {

    /***************************************************************************
     * Define a couple of frequently needed variables              *
     ***************************************************************************/

    word_width   = rae_->config.word_representation_size;
    approx_width = rae_->config.approx_width;
    dict_size    = rae_->getDictSize();

    /***************************************************************************
     *           Define access vectors to the various gradient parts           *
     ***************************************************************************/

    data = new Real[n];
    Real *ptr = data;

      new (&grad_D) WeightMatrixType(ptr, dict_size, word_width);
      grad_D.setZero();
      ptr += rae_->theta_D_size_;
      for (int i=0; i < dict_size; ++i)
      {
        grad_U.push_back(WeightMatrixType(ptr, word_width, approx_width));
        grad_U.back().setZero();
        ptr += word_width*approx_width;
      }
      for (int i=0; i < dict_size; ++i)
      {
        grad_V.push_back(WeightMatrixType(ptr, approx_width, word_width));
        grad_V.back().setZero();
        ptr += word_width*approx_width;
      }
      new (&grad_W) WeightMatrixType(ptr, dict_size, word_width);
      grad_W.setZero();
      ptr += rae_->theta_W_size_;
      new (&grad_A) WeightVectorType(ptr, dict_size);
      grad_A.setZero();
      ptr += dict_size;
      new (&grad_Wd) WeightVectorType(ptr, rae_->theta_Wd_size_);
      grad_Wd.setZero();
      ptr += rae_->theta_Wd_size_;
      new (&grad_Wf) WeightVectorType(ptr, rae_->theta_Wf_size_);
      grad_Wf.setZero();
      ptr += rae_->theta_Wf_size_;
      new (&grad_Wl) WeightVectorType(ptr, rae_->theta_Wl_size_);
      grad_Wl.setZero();
      ptr += rae_->theta_Wl_size_;

    assert (data + n == ptr);

    new (&weights) WeightVectorType(data,n);
  }


Backpropagator::~Backpropagator ()
{
  delete [] data;
}

int Backpropagator::backPropagateLbl(int i, VectorReal& x)
{
  SingleProp* propagator = nullptr;
  if(rae_->config.tree == TREE_CCG or rae_->config.tree == TREE_STANFORD)
    propagator = new SingleProp(rae_,model.corpus[i],model.beta,model.bools);
  else
    assert(false);
  propagator->forwardPropagate(true);
  propagator->setToD(x,0);
  int correct = propagator->backPropagate(true);

#pragma omp critical
  {
    if(model.corpus[i].value==0)
    {
      is_a_zero += propagator->getJointNodes();
      zero_should_be_zero += propagator->getClassCorrect();
      one_should_be_zero += (propagator->getJointNodes() - propagator->getClassCorrect());
    }
    else
    {
      is_a_one += propagator->getJointNodes();
      one_should_be_one += propagator->getClassCorrect();
      zero_should_be_one += (propagator->getJointNodes() - propagator->getClassCorrect());
    }

    error_ += propagator->getLblError();
    correctly_classified_sent += correct;
    setGradients_(propagator,i);
  }
  delete propagator;
  return correct;
}


void Backpropagator::backPropagateRae(int i, VectorReal& x)
{
  assert(false);
  SingleProp* propagator = nullptr;
  if(rae_->config.tree == TREE_CCG or rae_->config.tree == TREE_STANFORD)
    propagator = new SingleProp(rae_,model.corpus[i],model.beta,model.bools);
  else
    assert(false);
  propagator->forwardPropagate(false);
  propagator->setToD(x,0);
  propagator->backPropagate(false);

#pragma omp critical
  {
    error_ += propagator->getRaeError();
    setGradients_(propagator,i);
  }
  delete propagator;
}

void Backpropagator::normalize(int type) { // type: 0=rae, 1=lbl
  if (type == 0 or type == 2)
    return;

  float norm = 1.0;

  if (type == 0)      norm = count_nodes_ - count_words_;
  else if (type == 1) norm = max(1,model.to - model.from);

  error_ /= norm;
  grad_D /= norm;
  for (int x=0; x<dict_size; ++x) grad_U[x] /= norm;
  for (int x=0; x<dict_size; ++x) grad_V[x] /= norm;
  grad_W /= norm;
  grad_A /= norm;
  grad_Wd /= norm;
  grad_Wf /= norm;
  grad_Wl /= norm;
}

void Backpropagator::printInfo() {
  cout << "  " << correctly_classified_sent << "/" << model.to - model.from << " ";
  cout << "Z: " << is_a_zero << ": (" << zero_should_be_zero << " / " << one_should_be_zero << ")";
  cout << "O: " << is_a_one << ": (" << one_should_be_one << " / " << zero_should_be_one << ")";
  cout << endl;
}


void Backpropagator::setGradients_(SingleProp* propagator, int i) {

  count_words_ += propagator->getSentLength();
  count_nodes_ += propagator->getNodesLength();

  if (model.bools.Wd)  grad_Wd += propagator->getWdGradient();
  if (model.bools.Wf)  grad_Wf += propagator->getWfGradient();
  if (model.bools.Wl)  grad_Wl += propagator->getWlGradient();
  if (model.bools.D)
  {
    auto tmpD = propagator->getDGradients();
    for (size_t k = 0; k < model.corpus[i].words.size(); ++k)
      grad_D.row(model.corpus[i].words[k]) += tmpD[k];
  }
  if (model.bools.U)
  {
    auto tmpU = propagator->getUGradients();
    for (size_t k = 0; k < model.corpus[i].words.size(); ++k)
      grad_U[model.corpus[i].words[k]] += tmpU[k];
  }
  if (model.bools.V)
  {
    auto tmpV = propagator->getVGradients();
    for (size_t k = 0; k < model.corpus[i].words.size(); ++k)
      grad_V[model.corpus[i].words[k]] += tmpV[k];
  }
  if (model.bools.W)
  {
    auto tmpW = propagator->getWGradients();
    for (size_t k = 0; k < model.corpus[i].words.size(); ++k)
      grad_W.row(model.corpus[i].words[k]) += tmpW[k];
  }
  if (model.bools.A)
  {
    auto tmpA = propagator->getAGradients();
    for (size_t k = 0; k < model.corpus[i].words.size(); ++k)
      grad_A[model.corpus[i].words[k]] += tmpA[k][0];

  }
}

WeightVectorType Backpropagator::dump() { return weights; }
lbfgsfloatval_t  Backpropagator::getError() { return error_; }

}
