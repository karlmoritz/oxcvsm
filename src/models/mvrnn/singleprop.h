// File: singleprop.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 03-01-2013
// Last Update: Tue 01 Oct 2013 07:10:22 PM BST

/*------------------------------------------------------------------------
 * Description: <DESC> 
 * 
 *------------------------------------------------------------------------
 * History:
 * TODO:    
 *========================================================================
 */

#ifndef SINGLEPROP_H_59ULLTMO
#define SINGLEPROP_H_59ULLTMO

#include "../../common/shared_defs.h"
#include "../../common/singlepropbase.h"
#include "recursive_autoencoder.h"

namespace mvrnn
{

class SingleProp : public SinglePropBase
{

public:
  SingleProp (RecursiveAutoencoder* rae, const TrainingInstance &t, Real beta, Bools updates);

  virtual ~SingleProp ();
  void forwardPropagate(bool updateWcat);
  int backPropagate(bool updateWcat);                   // Backprop using the standard error signals (label/reconstruction)
  int backPropagateBi(VectorReal word, bool updateBi);  // Backprop given a word as the error signal on top
  int evaluateSentence();

  Real getLblError();
  Real getRaeError();

  WeightVectorsType   getDGradients();
  WeightMatricesType  getUGradients();
  WeightMatricesType  getVGradients();
  WeightVectorsType   getWGradients();
  WeightVectorsType   getAGradients();

  WeightVectorType getWdGradient();
  WeightVectorType getWfGradient();
  WeightVectorType getWlGradient();
  
  int getClassCorrect();
  int getJointNodes(); 

  int getClassZero();
  int getSentLength();
  int getNodesLength();

  void setDynamic(WeightVectorType& dynamic, int mode=0);

  void setToD(VectorReal& x, int i);

private:

  void encodeInputs(int i, int child0, int child1, int rule, int rc0, int rc1, bool updateWcat);
  void encodeSingular(int i, int child0, int rule, int rc0, bool updateWcat);
  int  applyLabel(int parent, bool updateWcat, Real beta);
  void backpropInputs(int node, int child0, int child1, int rule, int rc0, int rc1, bool updateWcat);
  void backpropWord(int node, int sent_pos);
  void backpropBi(int node, VectorReal word);

  Real beta;

  RecursiveAutoencoder* rae_;
  const CCGInstance& instance_;

  int sent_length;
  int nodes_length;

  Bools updates;

  // Inputs -> F-stages -> F
  // Inputs -> D-stages -> D
  // Inputs -> alpha

  // Values (forward propagation).
  // D0,D1 are intermediate stages for D etc.
  WeightVectorsType  D0;
  WeightVectorsType  D1;
  WeightVectorsType  D;
  WeightMatricesType F0;
  WeightMatricesType F1;
  WeightMatricesType F;
  WeightVectorsType  A;

  // Deltas (backpropagation)
  WeightVectorsType  Delta_D;
  WeightMatricesType Delta_F;
  WeightVectorsType  Delta_A;

  // Gradients
  WeightVectorsType     D_grad;
  WeightMatricesType    U_grad;
  WeightMatricesType    V_grad;
  WeightVectorsType     W_grad;
  WeightVectorsType     A_grad;
  WeightMatricesType    Wd0_grad;
  WeightMatricesType    Wd1_grad;
  WeightMatricesType    Wf0_grad;
  WeightMatricesType    Wf1_grad;
  WeightMatricesType    Wl_grad;

  // Thetas for W gradients for easy access
  WeightVectorType      Theta;
  WeightVectorType      Theta_Wd;
  WeightVectorType      Theta_Wf;
  WeightVectorType      Theta_Wl;

  Real class_error_;  // cost function for classification


  Real* w_data;   // word gradients
  Real* g_data;   // weight gradients
  Real* m_data;   // embeddings and deltas

  int   classified_correctly_;
  int   classified_wrongly_;

};

}

#endif /* end of include guard: SINGLEPROP_H_59ULLTMO */
