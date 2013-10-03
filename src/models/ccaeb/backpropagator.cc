// File: backpropagator.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-04-2013
// Last Update: Thu 03 Oct 2013 11:38:39 AM BST

/*------------------------------------------------------------------------
 * Description: <DESC>
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================
 */

#include "backpropagator.h"

namespace ccaeb
{

Backpropagator::Backpropagator (RecursiveAutoencoder* rae, Model &model,int n) : rae_(rae), model(model),
  grad_D(0,0,0), grad_Wd(0,0), grad_Wdr(0,0), grad_Bd(0,0), grad_Bdr(0,0),
  grad_Wl(0,0), grad_Bl(0,0), weights(0,0), correctly_classified_sent(0),
  zero_should_be_one(0), zero_should_be_zero(0), one_should_be_zero(0),
  one_should_be_one(0), is_a_zero(0), is_a_one(0), error_(0.0), count_nodes_(0),
  count_words_(0)
  {

    /***************************************************************************
     * Define a couple of frequently needed variables              *
     ***************************************************************************/

    word_width = rae_->config.word_representation_size;
    dict_size  = rae_->getDictSize();

    /***************************************************************************
     *           Define access vectors to the various gradient parts           *
     ***************************************************************************/

    data = new Real[n];
    Real *ptr = data;

      new (&grad_D) WeightMatrixType(ptr, dict_size, word_width);
      grad_D.setZero();
      ptr += rae_->theta_D_size_;
      new (&grad_Wd) WeightVectorType(ptr, rae_->theta_Wd_size_);
      grad_Wd.setZero();
      ptr += rae_->theta_Wd_size_;
      new (&grad_Wdr) WeightVectorType(ptr, rae_->theta_Wdr_size_);
      grad_Wdr.setZero();
      ptr += rae_->theta_Wdr_size_;
      new (&grad_Bd) WeightVectorType(ptr, rae_->theta_Bd_size_);
      grad_Bd.setZero();
      ptr += rae_->theta_Bd_size_;
      new (&grad_Bdr) WeightVectorType(ptr, rae_->theta_Bdr_size_);
      grad_Bdr.setZero();
      ptr += rae_->theta_Bdr_size_;
      new (&grad_Wl) WeightVectorType(ptr, rae_->theta_Wl_size_);
      grad_Wl.setZero();
      ptr += rae_->theta_Wl_size_;
      new (&grad_Bl) WeightVectorType(ptr, rae_->theta_Bl_size_);
      grad_Bl.setZero();
      ptr += rae_->theta_Bl_size_;

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
    count_words_ += propagator->getSentLength();
    count_nodes_ += propagator->getNodesLength();

    if (model.bools.Wd)   grad_Wd  += propagator->getWdGradient();
    if (model.bools.Wdr)  grad_Wdr += propagator->getWdrGradient();
    if (model.bools.Bd)   grad_Bd  += propagator->getBdGradient();
    if (model.bools.Bdr)  grad_Bdr += propagator->getBdrGradient();
    if (model.bools.Wl)   grad_Wl  += propagator->getWlGradient();
    if (model.bools.Bl)   grad_Bl  += propagator->getBlGradient();
    if (model.bools.D)
    {
      auto tmpD = propagator->getDGradients();
      for (size_t k = 0; k < model.corpus[i].words.size(); ++k)
        grad_D.row(model.corpus[i].words[k]) += tmpD[k];
    }
  }
  delete propagator;
  return correct;
}


void Backpropagator::backPropagateRae(int i, VectorReal& x)
{
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
    count_words_ += propagator->getSentLength();
    count_nodes_ += propagator->getNodesLength();

    if (model.bools.Wd)   grad_Wd  += propagator->getWdGradient();
    if (model.bools.Wdr)  grad_Wdr += propagator->getWdrGradient();
    if (model.bools.Bd)   grad_Bd  += propagator->getBdGradient();
    if (model.bools.Bdr)  grad_Bdr += propagator->getBdrGradient();
    if (model.bools.D)
    {
      auto tmpD = propagator->getDGradients();
      for (size_t k = 0; k < model.corpus[i].words.size(); ++k)
        grad_D.row(model.corpus[i].words[k]) += tmpD[k];
    }
  }

  delete propagator;
}

void Backpropagator::normalize(int type) { // type: 0=rae, 1=lbl

  float norm = 1.0;

  if (type == 0)      norm = count_nodes_ - count_words_;
  else if (type == 1) norm = max(1,model.to - model.from);

  error_ /= norm;
  grad_D /= norm;
  grad_Wd /= norm;
  grad_Wdr /= norm;
  grad_Bd /= norm;
  grad_Bdr /= norm;
  grad_Wl /= norm;
  grad_Bl /= norm;

}

void Backpropagator::printInfo() {
  cout << "  " << correctly_classified_sent << "/" << model.to - model.from << " ";
  cout << "Z: " << is_a_zero << ": (" << zero_should_be_zero << " / " << one_should_be_zero << ")";
  cout << "O: " << is_a_one << ": (" << one_should_be_one << " / " << zero_should_be_one << ")";
  cout << endl;
}

WeightVectorType Backpropagator::dump() { return weights; }
lbfgsfloatval_t  Backpropagator::getError() { return error_; }

}
