// File: recursive_autoencoder.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 02-01-2013
// Last Update: Thu 03 Oct 2013 11:33:27 AM BST

#ifndef RECURSIVE_AUTOENCODER_H_0HFOJXLA
#define RECURSIVE_AUTOENCODER_H_0HFOJXLA

#include "shared_defs.h"
#include "config.h"
#include "dictionary.h"
#include "senna.h"
#include "reindex_dict.h"

#include <lbfgs.h>


class SinglePropBase;
//class FinetuneClassifier;
class BackpropagatorBase;

class RecursiveAutoencoderBase
{

public:
  RecursiveAutoencoderBase(const ModelData& config, Lambdas L);
  RecursiveAutoencoderBase (const ModelData& config);
  virtual ~RecursiveAutoencoderBase ();
  virtual RecursiveAutoencoderBase* cloneEmpty () = 0;

  void finalizeDictionary (bool random_init=true); // To be called once the dictionary has been added
  void finalizeSpecific   (Real* theta_address); // To be called once the dictionary has been added
  void initFromWithDict(RecursiveAutoencoderBase& rae, std::map<LabelID,LabelID> n2o_map);

  virtual Real getLambdaCost(Bools bl) = 0;
  virtual void addLambdaGrad(Real* theta_data, Bools bl) = 0;

  void setLambdas(Lambdas l);
  void debugSize(int count);

  //const Dictionary& getDictionary() const { return dict_; }
  const Dictionary& getDictionary() const { return dict_; }
  Dictionary& getDictionary() { return dict_; }
  int getDictSize() { return dict_.num_labels(); }

  void averageUnknownWord();
  int getThetaSize();

  virtual void setIncrementalCounts(Counts &counts, Real *&vars, int &number) = 0;

  virtual BackpropagatorBase* getBackpropagator(Model &model, int n, int type) = 0; // type: rae=0, lbl=1, bi=2
  virtual SinglePropBase*     getSingleProp(const TrainingInstance &t, Real beta, Bools updates) = 0;

  virtual void init(bool init_words, bool create_new_theta=true) = 0;

  /***************************************************************************
   *                                Variables                                *
   ***************************************************************************/

  ModelData   config;
  Dictionary            dict_;
protected:

  Lambdas               lambdas;

  WeightMatrixType      D;  // Domain    (vector) (nx1)
  WeightVectorType      Theta_Full;
  WeightVectorType      Theta_D;

  Real*                 theta_;     // Pointer to all weights in my model (including embeddings)
  Real*                 theta_D_;

  int                   theta_size_;
  int                   theta_D_size_;

public:

  friend class Senna;
  friend void setVarsAndNumber(Real *&vars, int &number_vars, Model &model);
  friend int main(int argc, char **argv);

  friend RecursiveAutoencoderBase* reindex_dict(RecursiveAutoencoderBase& rae, TrainingCorpus& trainC, TrainingCorpus& testC);
  friend class boost::serialization::access;
  template<class Archive>
    void save(Archive& ar, const unsigned version) const {
      ar & dict_;
      ar & config;
      ar & boost::serialization::make_array(theta_, theta_size_);
    }

  template<class Archive>
    void load(Archive& ar, const unsigned version) {
      ar & dict_;
      ar & config;
      init(false); // initialize arrays and make space for theta_
      ar & boost::serialization::make_array(theta_, theta_size_);
    }
  BOOST_SERIALIZATION_SPLIT_MEMBER()

};

#endif /* end of include guard: RECURSIVE_AUTOENCODER_H_0HFOJXLA */

