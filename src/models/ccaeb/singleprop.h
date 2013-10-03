// File: singleprop.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 03-01-2013
// Last Update: Tue 01 Oct 2013 07:09:26 PM BST

/*------------------------------------------------------------------------
 * Description: <DESC> 
 * 
 *------------------------------------------------------------------------
 * History:
 * TODO:    
 *========================================================================
 */


#ifndef CCG_PROP_H_C74NY809
#define CCG_PROP_H_C74NY809

#include "../../common/shared_defs.h"
#include "../../common/singlepropbase.h"
#include "recursive_autoencoder.h"

namespace ccaeb
{

class SingleProp : public SinglePropBase
{

public:
  SingleProp (RecursiveAutoencoder* rae, const TrainingInstance &t, Real beta, Bools updates);

  virtual ~SingleProp ();
  void forwardPropagate(bool updateWcat);
  int backPropagate(bool updateWcat);
  int backPropagateBi(VectorReal word, bool updateBi);
  int evaluateSentence();

  Real getLblError();
  Real getRaeError();

  WeightVectorsType getDGradients();
  WeightVectorType  getWdGradient();
  WeightVectorType  getWdrGradient();
  WeightVectorType  getBdGradient(); 
  WeightVectorType  getBdrGradient();
  WeightVectorType  getWlGradient(); 
  WeightVectorType  getBlGradient(); 

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

  // Values (forward propagation).
  WeightVectorsType  D;
  WeightVectorsType  D_unnorm;
  WeightVectorsType  R0;
  WeightVectorsType  R1;

  // Deltas (backpropagation)
  WeightVectorsType  Delta_D;

  // Gradients
  WeightVectorsType     D_grad;

  WeightMatricesType    Wd0_grad;
  WeightMatricesType    Wd1_grad;
  WeightMatricesType    Wdr0_grad;
  WeightMatricesType    Wdr1_grad;

  WeightVectorsType     Bd0_grad;      // Bias weights
  WeightVectorsType     Bdr0_grad;     // Bias weights for reconstruction
  WeightVectorsType     Bdr1_grad;     // Bias weights for reconstruction

  WeightMatricesType    Wl_grad;
  WeightVectorsType     Bl_grad;

  // Thetas for W gradients and biases for easy access
  WeightVectorType      Theta;
  WeightVectorType      Theta_Wd;
  WeightVectorType      Theta_Wdr;
  WeightVectorType      Theta_Bd;
  WeightVectorType      Theta_Bdr;

  WeightVectorType      Theta_Wl;
  WeightVectorType      Theta_Bl;

  Real class_error_;  // cost function for classification
  Real tree_error_;  // cost function for classification

  Real* w_data;   // word gradients
  Real* m_data;   // embeddings and deltas (tmp storage)
  Real* g_data;   // weight gradients

  int   classified_correctly_;
  int   classified_wrongly_;

};

}

#endif /* end of include guard: CCG_PROP_H_C74NY809 */
