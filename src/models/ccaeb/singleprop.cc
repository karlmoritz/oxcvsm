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
 * TODO:    Return T-vector for BFGS/Gradient Ascent + Separate Error and
 * Gradient Calculation and do things
 *========================================================================
 */

#include <cmath>

#include "singleprop.h"
#include "../../common/fast_math.h"

namespace ccaeb
{

SingleProp::SingleProp(RecursiveAutoencoder* rae, const TrainingInstance &t, Real beta=1, Bools updates=Bools())
  : beta(beta), rae_(rae), instance_(static_cast<const CCGInstance&>(t)), updates(updates),
  Theta(0,0), Theta_Wd(0,0), Theta_Wdr(0,0), Theta_Bd(0,0), Theta_Bdr(0,0), Theta_Wl(0,0), Theta_Bl(0,0),
  class_error_(0.0), tree_error_(0.0), classified_correctly_(0), classified_wrongly_(0)
  {

    sent_length = int(instance_.words.size());
    nodes_length = int(instance_.rule.size());
    int word_width = rae_->config.word_representation_size;
    int label_width = rae_->config.label_class_size;

    /***************************************************************************
     *      Create data fields for temporary storage in this propagation       *
     ***************************************************************************/

    // embedding, embedding_unnormalised, reconstruction left and right, delta
    int m_data_size = nodes_length * (
        5 * word_width                    // D, D_unnorm, R0, R1, dD
        ); 

    m_data = new Real[m_data_size];
    Real* ptr = m_data;

    new (&Theta) WeightVectorType(m_data, m_data_size);
    Theta.setZero();

    for (auto i=0; i<nodes_length; ++i) {
      D_unnorm.push_back(WeightVectorType(ptr, word_width));
      ptr += word_width;
      D.push_back(WeightVectorType(ptr, word_width));
      ptr += word_width;
      R0.push_back(WeightVectorType(ptr, word_width));
      ptr += word_width;
      R1.push_back(WeightVectorType(ptr, word_width));
      ptr += word_width;
      Delta_D.push_back(WeightVectorType(ptr, word_width));
      ptr += word_width;
    }

    assert(ptr == m_data+m_data_size); 

    for (auto i=0; i<nodes_length; ++i)
    {
      int node = instance_.nodes[i];
      if (node != -1)
      {
        D_unnorm[i] = rae_->D.row(instance_.words[node]);
        D[i] = rae_->D.row(instance_.words[node]); //.normalized(); 
        // !!!!!!!! FIXME: would need to modify gradients if this was normalized
      }
    }


    /***************************************************************************
     *             Create data fields for word embedding gradients             *
     ***************************************************************************/

    if (updates.D)
    {
      int we_grads_size = sent_length*(
          word_width                              // Domain
          );
      w_data = new Real[we_grads_size];
      new (&Theta) WeightVectorType(w_data, we_grads_size);
      Theta.setZero();
      ptr = w_data;

      for (int p_left=0; p_left < sent_length; ++p_left)
      {
        D_grad.push_back(WeightVectorType(ptr, word_width));
        ptr += word_width;
      }
    }
    else
      w_data = nullptr;

    /***************************************************************************
     *                    Create data fields for gradients                     *
     ***************************************************************************/

    int g_data_size = rae_->theta_size_ - rae_->theta_D_size_;
    g_data = new Real[g_data_size];
    new (&Theta) WeightVectorType(g_data, g_data_size);
    Theta.setZero();

    ptr = g_data;

    new (&Theta_Wd) WeightVectorType(ptr, rae_->theta_Wd_size_); ptr += rae_->theta_Wd_size_;
    new (&Theta_Wdr) WeightVectorType(ptr, rae_->theta_Wdr_size_); ptr += rae_->theta_Wdr_size_;
    new (&Theta_Bd) WeightVectorType(ptr, rae_->theta_Bd_size_); ptr += rae_->theta_Bd_size_;
    new (&Theta_Bdr) WeightVectorType(ptr, rae_->theta_Bdr_size_); ptr += rae_->theta_Bdr_size_;
    new (&Theta_Wl) WeightVectorType(ptr, rae_->theta_Wl_size_); ptr += rae_->theta_Wl_size_;
    new (&Theta_Bl) WeightVectorType(ptr, rae_->theta_Bl_size_); ptr += rae_->theta_Bl_size_;
    assert (ptr == g_data + g_data_size);

    ptr = g_data;
    // Initialize Domain Matrices
    for (auto i=0; i<num_ccg_rules; ++i) {
      Wd0_grad.push_back(WeightMatrixType(ptr, word_width, word_width));
      ptr += word_width*word_width;
      Wd1_grad.push_back(WeightMatrixType(ptr, word_width, word_width));
      ptr += word_width*word_width;
    }

    // Initialize Domain Reconstruction Matrices
    for (auto i=0; i<num_ccg_rules; ++i) {
      Wdr0_grad.push_back(WeightMatrixType(ptr, word_width, word_width));
      ptr += word_width*word_width;
      Wdr1_grad.push_back(WeightMatrixType(ptr, word_width, word_width));
      ptr += word_width*word_width;
    }

    // Initialize Domain Biases
    for (auto i=0; i<num_ccg_rules; ++i) {
      Bd0_grad.push_back(WeightVectorType(ptr, word_width));
      ptr += word_width;
    }

    // Initialize Domain Reconstruction Biases
    for (auto i=0; i<num_ccg_rules; ++i) {
      Bdr0_grad.push_back(WeightVectorType(ptr, word_width));
      ptr += word_width;
      Bdr1_grad.push_back(WeightVectorType(ptr, word_width));
      ptr += word_width;
    }

    Wl_grad.push_back(WeightMatrixType(ptr, label_width, word_width));
    ptr += label_width*word_width;

    Bl_grad.push_back(WeightVectorType(ptr, label_width));
    ptr += label_width;

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
    int rule = instance_.rule[i];
    if(child1 >= 0)
    {
      int rc0 = instance_.cat[child0];
      int rc1 = instance_.cat[child1];
      encodeInputs(i, child0, child1, rule, rc0, rc1, use_lbl_error);
    }
    else if (child0 >= 0)
    {
      int rc0 = instance_.cat[child0];
      encodeSingular(i, child0, rule, rc0, use_lbl_error);
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

    int rc0 = -1;
    int rc1 = -1;

    if(child0 >= 0)
      rc0 = instance_.cat[child0];
    if(child1 >= 0)
      rc1 = instance_.cat[child1];

    // only on the top? not in Socher
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
      rc1 = 0;
    }

    if (rule == LEAF)
    {
      if (updates.D)
        backpropWord(i, instance_.nodes[i]);
    }
    else
      backpropInputs(i, child0, child1, rule, rc0, rc1, use_lbl_error);
  }
  //Theta = Theta / nodes_length; // normalise gradients by number of nodes
  return (classified_correctly_ > classified_wrongly_) ? 1 : 0;
}                                                              



/******************************************************************************
 *                        Actual composition functions                        *
 ******************************************************************************/


void SingleProp::encodeInputs(int node, int child0, int child1, int rule, int rc0, int rc1, bool only_lbl_error)
{

  D_unnorm[node] = (rae_->Wd0[rule] * D[child0] + rae_->Wd1[rule] * D[child1] + rae_->Bd0[rule]).unaryExpr(std::ptr_fun(getTanh));
  D[node] = D_unnorm[node].normalized();

  if (not only_lbl_error)
  {
    R0[node] = (rae_->Wdr0[rule] * D[node] + rae_->Bdr0[rule]).unaryExpr(std::ptr_fun(getTanh));
    R1[node] = (rae_->Wdr1[rule] * D[node] + rae_->Bdr1[rule]).unaryExpr(std::ptr_fun(getTanh));
  }
}

void SingleProp::encodeSingular(int node, int child0, int rule, int rc0, bool use_lbl_error)
{                 
  // Let's skip single rules for now. We ignore most of CCG anyway...
  D[node] = D[child0];
  D_unnorm[node] = D_unnorm[child0];
}

/******************************************************************************
 *                      Actual backpropagation functions                      *
 ******************************************************************************/


int SingleProp::applyLabel(int node, bool use_lbl_error, Real beta)
{

  ArrayReal label_pred = (
      rae_->Wl[0] * D[node] + rae_->Bl[0]
      ).unaryExpr(std::ptr_fun(getSigmoid)).array();

  ArrayReal label_correct(rae_->config.label_class_size);
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

  if(correct == 1)  classified_correctly_ += 1;
  else              classified_wrongly_   += 1;

  Wl_grad[0] +=    rae_->lambdas.alpha_lbl * label_delta.matrix() * D[node].transpose();
  Bl_grad[0] +=    rae_->lambdas.alpha_lbl * label_delta.matrix();
  Delta_D[node] += rae_->lambdas.alpha_lbl * rae_->Wl[0].transpose() * label_delta.matrix();
  class_error_ +=  rae_->lambdas.alpha_lbl * lbl_error; 

  return correct;
}


void SingleProp::backpropInputs(int node, int child0, int child1, int rule, int rc0, int rc1, bool no_rae_error)
{
  if (child0 == child1)
  {
    Delta_D[child0] = Delta_D[node];
    return;
  }

  VectorReal _D_unnorm = D_unnorm[node];

  VectorReal delta_here;
  VectorReal delta_tree = Delta_D[node].matrix();

  // Initialize to zero
  VectorReal delta_child0(D[node].rows());
  VectorReal delta_child1(D[node].rows());
  delta_child0.setZero();
  delta_child1.setZero();

  if (no_rae_error)
    delta_here = tanh_p(_D_unnorm) * delta_tree; 
  else
  {
    VectorReal R0_norm = R0[node].normalized();
    VectorReal R1_norm = R1[node].normalized();
    VectorReal _R0_unnorm = R0[node];
    VectorReal _R1_unnorm = R1[node];

    VectorReal delta_at_R0 = tanh_p(_R0_unnorm) * (rae_->lambdas.alpha_rae * (R0_norm - D[child0]));
    VectorReal delta_at_R1 = tanh_p(_R1_unnorm) * (rae_->lambdas.alpha_rae * (R1_norm - D[child1]));

    tree_error_ += (0.5 * rae_->lambdas.alpha_rae * (R0_norm - D[child0]).array().pow(2)).sum();
    tree_error_ += (0.5 * rae_->lambdas.alpha_rae * (R1_norm - D[child1]).array().pow(2)).sum();

    MatrixReal delta_from_R0 = rae_->Wdr0[rule].transpose() * delta_at_R0;
    MatrixReal delta_from_R1 = rae_->Wdr1[rule].transpose() * delta_at_R1;

    delta_here = 
      tanh_p(_D_unnorm) *
      (
       delta_from_R0 +
       delta_from_R1 +
       delta_tree
      );

    delta_child0 -= rae_->lambdas.alpha_rae * (R0_norm - D[child0]);
    delta_child1 -= rae_->lambdas.alpha_rae * (R1_norm - D[child1]);

    Bdr0_grad[rule] += delta_at_R0;
    Bdr1_grad[rule] += delta_at_R1;

    Wdr0_grad[rule] += delta_at_R0 * D[node].transpose();
    Wdr1_grad[rule] += delta_at_R1 * D[node].transpose();
  }


  delta_child0 += rae_->Wd0[rule].transpose() * delta_here;
  delta_child1 += rae_->Wd1[rule].transpose() * delta_here; 

  Bd0_grad[rule] += delta_here;

  Wd0_grad[rule] += delta_here * D[child0].transpose();
  Wd1_grad[rule] += delta_here * D[child1].transpose();

  Delta_D[child0] += delta_child0;
  Delta_D[child1] += delta_child1;

}

void SingleProp::backpropWord(int node, int sent_pos)
{
  D_grad[sent_pos] = Delta_D[node];
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

    int rc0 = -1;
    int rc1 = -1;

    if(child0 >= 0)
      rc0 = instance_.cat[child0];
    if(child1 >= 0)
      rc1 = instance_.cat[child1];

    // Get the parallel error (currently only at the top node)
    if (i == 0 && updateBi == true)
      backpropBi(i,word);

    if (child1 == -1)
    {
      child1 = child0;
      rc1 = 0;
    }

    if (rule == LEAF)
    {
      if (updates.D)
        backpropWord(i, instance_.nodes[i]);
    }
    else
      backpropInputs(i, child0, child1, 0, rc0, rc1, updateBi);
  }
  //Theta = Theta / nodes_length; // normalise gradients by number of nodes
  return (classified_correctly_ > classified_wrongly_) ? 1 : 0;
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
Real SingleProp::getRaeError() { return tree_error_; }

WeightVectorsType SingleProp::getDGradients()  { return D_grad;    }
WeightVectorType  SingleProp::getWdGradient()  { return Theta_Wd;  }
WeightVectorType  SingleProp::getWdrGradient() { return Theta_Wdr; }
WeightVectorType  SingleProp::getBdGradient()  { return Theta_Bd;  }
WeightVectorType  SingleProp::getBdrGradient() { return Theta_Bdr; }
WeightVectorType  SingleProp::getWlGradient()  { return Theta_Wl; }
WeightVectorType  SingleProp::getBlGradient()  { return Theta_Bl; }

int SingleProp::getClassCorrect() { return classified_correctly_; }
int SingleProp::getJointNodes() { return nodes_length; }

void SingleProp::setToD(VectorReal& x, int i) { x = D[i]; }

}
