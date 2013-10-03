// File: singlepropbase.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 03-01-2013
// Last Update: Tue 01 Oct 2013 07:09:25 PM BST

/*------------------------------------------------------------------------
 * Description: <DESC> 
 * 
 *------------------------------------------------------------------------
 * History:
 * TODO:    
 *========================================================================
 */

#ifndef SINGLEPROPBASE_H_EAQWCBK8
#define SINGLEPROPBASE_H_EAQWCBK8

#include "shared_defs.h"
#include "recursive_autoencoder.h"

class SinglePropBase
{

public:
  //SingleProp (RecursiveAutoencoder* rae, const TrainingInstance &t, Real beta, Bools updates);

  virtual ~SinglePropBase () {};
  virtual void forwardPropagate(bool updateWcat) = 0;
  virtual int evaluateSentence() = 0;

  virtual void setDynamic(WeightVectorType& dynamic, int mode=0) = 0;
};

#endif /* end of include guard: SINGLEPROPBASE_H_EAQWCBK8 */
