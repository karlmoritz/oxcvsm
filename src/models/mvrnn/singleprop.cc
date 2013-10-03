// File: singleprop.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 13-01-2013
// Last Update: Tue 01 Oct 2013 07:05:44 PM BST

/*------------------------------------------------------------------------
 * Description: Builds on the RecursiveAutoencoder Class and provides
 * functions for forward and backpropagation as well as tree building
 *
 * Alternative tree-types should/could inherit from this class
 * 
 *------------------------------------------------------------------------
 * History:
 * TODO:    Further clean-up. Perhaps install hierarchies to inherit from 
 * more basic MVRNN / allow other models to inherit from here
 *========================================================================
 */

#include <cmath>

#include "singleprop.h"
#include "../../common/fast_math.h"
//#include "../../common/grammarrules.h"

namespace mvrnn
{

  SingleProp::SingleProp(RecursiveAutoencoder* rae, const TrainingInstance &t, Real beta=1, Bools updates=Bools())
: beta(beta), rae_(rae), instance_(static_cast<const CCGInstance&>(t)), updates(updates),
  Theta(0,0), Theta_Wd(0,0), Theta_Wf(0,0), Theta_Wl(0,0),
  class_error_(0.0), classified_correctly_(0), classified_wrongly_(0)
{
  sent_length = int(instance_.words.size());
  nodes_length = int(instance_.rule.size());

  int num_weight_types_ = 1;

  /***************************************************************************
   *      Create data fields for temporary storage in this propagation       *
   ***************************************************************************/

  int word_width = rae_->config.word_representation_size;
  int approx_width = rae_->config.approx_width;

  // embedding, embedding_unnormalised, reconstruction left and right, delta
  int m_data_size = nodes_length * (
      4 * word_width +                  // D0, D1, D, Delta_D
      4 * word_width*word_width   +     // F0, F1, F, Delta_F
      2                                 // A, Delta_A
      ); 

  m_data = new Real[m_data_size];
  Real* ptr = m_data;

  new (&Theta) WeightVectorType(m_data, m_data_size);
  Theta.setZero();

  for (auto i=0; i<nodes_length; ++i) {
    D0.push_back(WeightVectorType(ptr, word_width));
    ptr += word_width;
    D1.push_back(WeightVectorType(ptr, word_width));
    ptr += word_width;
    D.push_back(WeightVectorType(ptr, word_width));
    ptr += word_width;
    Delta_D.push_back(WeightVectorType(ptr, word_width));
    ptr += word_width;

    F0.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;
    F1.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;
    F.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;
    Delta_F.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;

    A.push_back(WeightVectorType(ptr, 1)); ptr += 1;
    Delta_A.push_back(WeightVectorType(ptr, 1)); ptr += 1;
  }

  assert(ptr == m_data+m_data_size); 

  for (auto i=0; i<nodes_length; ++i)
  {
    int node = instance_.nodes[i];
    if (node != -1)
    {
      D[i] = rae_->D.row(instance_.words[node]);
      //F[i] = Y+X; // doesn't work directly because diagonalWrapper acts funny
      F[i] = (rae_->U[instance_.words[node]]*rae_->V[instance_.words[node]]);
      F[i].diagonal() += rae_->W.row(instance_.words[node]);
      A[i] = rae_->A.row(instance_.words[node]);
    }
  }


  /***************************************************************************
   *             Create data fields for word embedding gradients             *
   ***************************************************************************/

  if (true) // assuming updates.F (U,V,W), updates.A
  {
    int we_grads_size = sent_length*(
        word_width +                            // Domain
        2*word_width*approx_width + word_width +  // Function (UV+diag)
        1                                       // Alpha
        );
    w_data = new Real[we_grads_size];
    new (&Theta) WeightVectorType(w_data, we_grads_size);
    Theta.setZero();
    ptr = w_data;

    for (int p_left=0; p_left < sent_length; ++p_left)
    {
      D_grad.push_back(WeightVectorType(ptr, word_width));
      ptr += word_width;
      U_grad.push_back(WeightMatrixType(ptr, word_width, approx_width));
      ptr += word_width*approx_width;
      V_grad.push_back(WeightMatrixType(ptr, approx_width, word_width));
      ptr += word_width*approx_width;
      W_grad.push_back(WeightVectorType(ptr, word_width));
      ptr += word_width;
      A_grad.push_back(WeightVectorType(ptr, 1));
      ptr += 1;
    }
  }
  else
    w_data = nullptr;

  /***************************************************************************
   *                    Create data fields for gradients                     *
   ***************************************************************************/

  int label_width = rae_->config.label_class_size;

  int g_Wd_size = num_weight_types_ * 2 * word_width * word_width;
  int g_Wf_size = num_weight_types_ * 2 * word_width * word_width;
  int g_Wl_size = label_width * word_width;
  int g_data_size = g_Wd_size + g_Wf_size + g_Wl_size;

  // Add some asserts here perhaps!

  g_data = new Real[g_data_size]();

  new (&Theta) WeightVectorType(g_data, g_data_size);
  Theta.setZero();
  new (&Theta_Wd) WeightVectorType(g_data, g_Wd_size);
  new (&Theta_Wf) WeightVectorType(g_data + g_Wd_size, g_Wf_size);
  new (&Theta_Wl) WeightVectorType(g_data + g_Wd_size + g_Wf_size, g_Wl_size);

  ptr = g_data;

  for (auto i=0; i<num_weight_types_; ++i) {
    Wd0_grad.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;
    Wd1_grad.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;
  }
  for (auto i=0; i<num_weight_types_; ++i) {
    Wf0_grad.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;
    Wf1_grad.push_back(WeightMatrixType(ptr, word_width, word_width));
    ptr += word_width*word_width;
  }
  for (auto i=0; i<1; ++i) {
    Wl_grad.push_back(WeightMatrixType(ptr, label_width, word_width));
    ptr += label_width*word_width;
  }

  assert(ptr== g_data+g_data_size); 

}

SingleProp::~SingleProp() {
  delete [] g_data;
  delete [] m_data;
  delete [] w_data;
}

void SingleProp::forwardPropagate(bool use_lbl_error)
{
  for (int i = nodes_length-1; i>= 0; --i)
  {
    int child0 = instance_.child0[i];
    int child1 = instance_.child1[i];
    //int rule = instance_.rule[i];
    if(child1 >= 0)
    {              
      //int rc0 = instance_.cat[child0];
      //int rc1 = instance_.cat[child1];
      encodeInputs(i, child0, child1, 0, 0, 0, use_lbl_error);
    }
    else if (child0 >= 0)
    {                   
      //int rc0 = instance_.cat[child0];
      encodeSingular(i, child0, 0, 0, use_lbl_error);
    }
  }
}

int SingleProp::evaluateSentence()
{
  for (int i = 0; i<nodes_length; ++i)
  {
    int child0 = instance_.child0[i];
    if (child0 == -1)
      applyLabel(i,true,1.0);
    else
      applyLabel(i,true,beta);
  }
  return (classified_correctly_ > classified_wrongly_) ? 1 : 0;
}


int SingleProp::backPropagate(bool use_lbl_error)
{

  for (int i = 0; i<nodes_length; ++i)
  {
    int child0 = instance_.child0[i];
    int child1 = instance_.child1[i];
    int rule = instance_.rule[i];

/*
 *    int rc0 = -1;
 *    int rc1 = -1;
 *
 *    if(child0 >= 0)
 *      rc0 = instance_.cat[child0];
 *    if(child1 >= 0)
 *      rc1 = instance_.cat[child1];
 */

    // only on the top? not in Socher 2012
    if (use_lbl_error)
    {
      if (child0 == -1)
        applyLabel(i,true,1.0);
      else
        applyLabel(i,true,beta);
    }

    if (child1 == -1)
    {
      child1 = child0;
    }

    if (rule == LEAF)
    {
      //if (updates.D)
        backpropWord(i, instance_.nodes[i]);
    }
    else
      backpropInputs(i, child0, child1, 0, 0, 0, use_lbl_error);
  }
  return (classified_correctly_ > classified_wrongly_) ? 1 : 0;
}                                                              



/******************************************************************************
 *                        Actual composition functions                        *
 ******************************************************************************/


void SingleProp::encodeInputs(int node, int child0, int child1, int rule, int rc0, int rc1, bool only_lbl_error)
{
  // Function combination
  F0[node] = F[child0];
  F1[node] = F[child1];
  F[node] = rae_->Wf0[rule] * F0[node] + rae_->Wf1[rule] * F1[node];

  // Domain combination
  D0[node] = F[child1] * D[child0];
  D1[node] = F[child0] * D[child1];
  auto X = (rae_->Wd0[rule] * D0[node] + rae_->Wd1[rule] * D1[node]);
  D[node] = (rae_->Wd0[rule] * D0[node] + rae_->Wd1[rule] * D1[node]).unaryExpr(std::ptr_fun(getTanh));
}

void SingleProp::encodeSingular(int node, int child0, int rule, int rc0, bool use_lbl_error)
{                 
  // Let's skip single rules for now. We ignore most of CCG anyway...
  D[node] = D[child0];
  F[node] = F[child0];
  //cout << "s " << D[node][0] << endl;
}


int SingleProp::applyLabel(int node, bool use_lbl_error, Real beta)
{

  VectorReal current = D[node];

  ArrayReal label_pred = (
      rae_->Wl[0] * current
      ).unaryExpr(std::ptr_fun(getSigmoid)).array();

  ArrayReal label_correct(rae_->config.label_class_size); // == 1
  assert(rae_->config.label_class_size == 1);
  label_correct[0] = instance_.value;

  // dE/dv * sigmoid'(net)
  ArrayReal label_delta = - beta * (label_correct - label_pred) * (1-label_pred) * (label_pred);
  Real lbl_error = 0.5 * beta * ((label_pred - label_correct) * (label_pred - label_correct)).sum();

  int correct = 0;
  if(
      (label_correct[0] == 0 && label_pred[0] <= 0.5)
      ||
      (label_correct[0] == 1 && label_pred[0] > 0.5)
    )
    correct = 1;

  if(correct == 1)
    classified_correctly_ += 1;
  else 
    classified_wrongly_ += 1;

  Wl_grad[0] += label_delta.matrix() * current.transpose();
  Delta_D[node] += rae_->Wl[0].transpose() * label_delta.matrix();
  class_error_ += lbl_error; 

  return correct;
}

void SingleProp::backpropInputs(int node, int child0, int child1, int rule, int rc0, int rc1, bool use_lbl_error)
{
  if (child0 == child1)
  {
    Delta_D[child0] = Delta_D[node];
    Delta_F[child0] = Delta_F[node];
    return;
  }

  VectorReal d  = D[node];
  VectorReal d0 = D0[node];
  VectorReal d1 = D1[node];
  MatrixReal f  = F[node];
  MatrixReal f0 = F0[node];
  MatrixReal f1 = F1[node];
  //Real       a  = A[node][0];

  VectorReal dc0 = D[child0];
  VectorReal dc1 = D[child1];
  MatrixReal fc0 = F[child0];
  MatrixReal fc1 = F[child1];

  VectorReal partialE_Z = (Delta_D[node].array() * (1 - (d.array()*d.array()))).matrix();

  Wd0_grad[rule] += partialE_Z * d0.transpose();
  Wd1_grad[rule] += partialE_Z * d1.transpose();

  Delta_D[child0] += F[child1].transpose() * (rae_->Wd0[rule].transpose() * partialE_Z);
  Delta_D[child1] += F[child0].transpose() * (rae_->Wd1[rule].transpose() * partialE_Z);

  MatrixReal partialE_F0 = rae_->Wf0[rule].transpose() * Delta_F[node];
  MatrixReal partialE_F1 = rae_->Wf1[rule].transpose() * Delta_F[node];

  Delta_F[child0] += partialE_F0;
  Delta_F[child1] += partialE_F1;

  Delta_F[child0] += (rae_->Wd1[rule].transpose() * partialE_Z) * dc1.transpose();
  Delta_F[child1] += (rae_->Wd0[rule].transpose() * partialE_Z) * dc0.transpose();

  Wf0_grad[rule] += Delta_F[node] * F[child0].transpose();
  Wf1_grad[rule] += Delta_F[node] * F[child1].transpose();

}

void SingleProp::backpropWord(int node, int sent_pos)
{
  if (updates.D)  D_grad[sent_pos] = Delta_D[node];
  if (updates.A)  A_grad[sent_pos] = Delta_A[node];

  // This part will need work! 
  int wordid = instance_.words[sent_pos];

  if (updates.U)
    U_grad[sent_pos] = Delta_F[node] * rae_->V[wordid].transpose(); 
  if (updates.V)
    V_grad[sent_pos] = rae_->U[wordid].transpose() * Delta_F[node]; 
  if (updates.W)
    W_grad[sent_pos] = Delta_F[node].diagonal();
}


/******************************************************************************
 *                    Special backprop for bilingual model                    *
 ******************************************************************************/

int SingleProp::backPropagateBi(VectorReal word, bool updateBi)
{
  // Assume updateBi is true for now. The functionality is otherwise contained
  // in the standard backprop. Will merge these two functions eventually
  assert (updateBi);

  for (int i = 0; i<nodes_length; ++i)
  {
    int child0 = instance_.child0[i];
    int child1 = instance_.child1[i];
    int rule = instance_.rule[i];

/*
 *    int rc0 = -1;
 *    int rc1 = -1;
 *
 *    if(child0 >= 0)
 *      rc0 = instance_.cat[child0];
 *    if(child1 >= 0)
 *      rc1 = instance_.cat[child1];
 */

    // Get the parallel error (currently only at the top node)
    if (i == 0 && updateBi == true)
      backpropBi(i,word);

    if (child1 == -1)
      child1 = child0;

    if (rule == LEAF)
    {
      //if (updates.D)
        backpropWord(i, instance_.nodes[i]);
    }
    else
      backpropInputs(i, child0, child1, 0, 0, 0, updateBi);
  }
  return 0;
  //return (classified_correctly_ > classified_wrongly_) ? 1 : 0;// pfft. no actual return here
}                                                              

void SingleProp::backpropBi(int node, VectorReal word)
{
  /*
   * Take a word and get the error (distance to own root word)
   */

  ArrayReal label_correct = word.array();
  ArrayReal label_pred = D[node].array();

  // Error: 0.5 (me - other)^2
  ArrayReal label_delta = (label_pred - label_correct);
  Real lbl_error = 0.5 * ((label_pred - label_correct) * (label_pred - label_correct)).sum();

  Delta_D[node] += rae_->lambdas.alpha_lbl * label_delta.matrix();
  class_error_ +=  rae_->lambdas.alpha_lbl * lbl_error; 
}


/******************************************************************************
 *                            Helper functions                                *
 ******************************************************************************/


int SingleProp::getSentLength()
{
  return sent_length;
}

int SingleProp::getNodesLength()
{
  return nodes_length;
}

void SingleProp::setDynamic(WeightVectorType& dynamic, int mode) 
{
  int embedding_width = rae_->config.word_representation_size;
  int shift = 0;
  if(mode >= 10)
  {
    shift = embedding_width;
    mode -= 10;
  }
  switch (mode)
  {
    case 0:
      // Set first half to the top vector, second average all nodes
      shift *= 2;
      dynamic.segment(shift,embedding_width) = D[0];
      for (auto i=0; i<nodes_length; ++i)
        dynamic.segment(shift+embedding_width,embedding_width) += D[i];
      dynamic.segment(shift+embedding_width,embedding_width) /= nodes_length;
      break;
    case 1:
      dynamic.segment(shift,embedding_width) = D[0];
      break;                                               
    case 2:
      for (auto i=0; i<nodes_length; ++i)
        dynamic.segment(shift,embedding_width) += D[i];
      dynamic.segment(shift,embedding_width) /= nodes_length;
      break;
    case 3:
      {
        int n = D[0].size();
        int pos = 0;
        for (auto i=0; i<nodes_length; ++i)
        {
          dynamic.segment(shift+pos,n) = D[i];
          pos += n;
        }
      }
      break;
    case 4:
      {
        shift *= 4;
        dynamic.segment(shift,embedding_width) += D[0];
        for (auto i=0; i<nodes_length; ++i)
          dynamic.segment(shift+embedding_width,embedding_width) += D[i];
        dynamic.segment(shift+embedding_width,embedding_width) /= nodes_length;
        int leaf_count = 0;
        int inner_count = 0;
        for (auto i=0; i<nodes_length; ++i)
        {
          if (instance_.rule[i] == LEAF)
          {
            leaf_count++;
            dynamic.segment(shift+2*embedding_width,embedding_width) += D[i];
          }
          else
          {
            inner_count++;
            dynamic.segment(shift+3*embedding_width,embedding_width) += D[i];
          }
        }
        dynamic.segment(shift+2*embedding_width,embedding_width) /= max(leaf_count,1);
        dynamic.segment(shift+3*embedding_width,embedding_width) /= max(inner_count,1);
      }
      break;
  }
}

Real SingleProp::getLblError() { return class_error_; }
Real SingleProp::getRaeError() { assert(false); return 0; }

WeightVectorsType   SingleProp::getDGradients() { return D_grad; }
WeightMatricesType  SingleProp::getUGradients() { return U_grad; }
WeightMatricesType  SingleProp::getVGradients() { return V_grad; }
WeightVectorsType  SingleProp::getWGradients() { return W_grad; }
WeightVectorsType    SingleProp::getAGradients() { return A_grad; }

WeightVectorType SingleProp::getWdGradient() { return Theta_Wd; }
WeightVectorType SingleProp::getWfGradient() { return Theta_Wf; }
WeightVectorType SingleProp::getWlGradient() { return Theta_Wl; }

int SingleProp::getClassCorrect() { return classified_correctly_; }
int SingleProp::getJointNodes() { return nodes_length; }

void SingleProp::setToD(VectorReal& x, int i) { x = D[i]; }

}
