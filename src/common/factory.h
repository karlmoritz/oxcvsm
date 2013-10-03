// File: factory.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 08-05-2013
// Last Update: Tue 01 Oct 2013 07:09:23 PM BST

/*------------------------------------------------------------------------
 * Description: <DESC> 
 * 
 *------------------------------------------------------------------------
 * History:
 * TODO:    
 *========================================================================
 */

#ifndef FACTORY_H_UER4CNV8
#define FACTORY_H_UER4CNV8

#include "recursive_autoencoder.h"
#include "shared_defs.h"

class Factory
{
public:
  Factory ();
  virtual ~Factory () {};
  virtual RecursiveAutoencoder* makeRae(const ModelData& config, Lambdas l) = 0;
};

#endif /* end of include guard: FACTORY_H_UER4CNV8 */
