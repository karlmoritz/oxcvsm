// File: recursive_autoencoder.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 02-01-2013
// Last Update: Tue 01 Oct 2013 07:09:25 PM BST

/*------------------------------------------------------------------------
 * Description: <DESC> 
 * 
 *------------------------------------------------------------------------
 * History:
 * TODO:    
 *========================================================================
 */

#ifndef RECURSIVE_AUTOENCODER_H_JVBAVFOJ
#define RECURSIVE_AUTOENCODER_H_JVBAVFOJ

#include "../../common/recursive_autoencoder.h"
#include <lbfgs.h>


class SingleProp;
class FinetuneClassifier;

namespace ccaeb
{

class Backpropagator;

class RecursiveAutoencoder : public RecursiveAutoencoderBase
  { 

  public:
    RecursiveAutoencoder(const ModelData& config, Lambdas L);
    RecursiveAutoencoder (const ModelData& config);

    virtual ~RecursiveAutoencoder ();
    RecursiveAutoencoderBase* cloneEmpty ();

    Real getLambdaCost(Bools bl);
    void addLambdaGrad(Real* theta_data, Bools bl);

    void setIncrementalCounts(Counts &counts, Real *&vars, int &number);

    BackpropagatorBase* getBackpropagator(Model &model, int n, int type);
    SinglePropBase*     getSingleProp(const TrainingInstance &t, Real beta, Bools updates);

  private:
    void init(bool init_words, bool create_new_theta=true);

    /***************************************************************************
     *                                Variables                                *
     ***************************************************************************/

  public:
    //ModelData   config;

  private:
    //WeightMatrixType      D;  // Domain    (vector) (nx1)

    WeightMatricesType    Wd0;     // Domain composition (nx2n)
    WeightMatricesType    Wd1;     // Domain composition (nx2n)
    WeightMatricesType    Wdr0;     // Reconstr
    WeightMatricesType    Wdr1;     // Reconst

    WeightVectorsType     Bd0;     // Bias weights
    WeightVectorsType     Bdr0;     // Bias weights for reconstruction
    WeightVectorsType     Bdr1;     // Bias weights for reconstruction

    WeightMatricesType    Wl;     // Label (nxl)
    WeightVectorsType     Bl;     // Bias  (l)

    //WeightVectorType      Theta_Full;         // Theta vector over the whole model (including embeddings)

    //WeightVectorType      Theta_D;
    WeightVectorType      Theta_Wd;
    WeightVectorType      Theta_Wdr;
    WeightVectorType      Theta_Bd;
    WeightVectorType      Theta_Bdr;
    WeightVectorType      Theta_Wl;
    WeightVectorType      Theta_Bl;

    //Dictionary            dict_;
    //Real*                 theta_;     // Pointer to all weights in my model (including embeddings)
    //Real*                 theta_D_;
    Real*                 theta_Wd_;
    Real*                 theta_Wdr_;
    Real*                 theta_Bd_;
    Real*                 theta_Bdr_;
    Real*                 theta_Wl_;
    Real*                 theta_Bl_;

    //int         theta_size_;
    //int         theta_D_size_;
    int         theta_Wd_size_;
    int         theta_Wdr_size_;
    int         theta_Bd_size_;
    int         theta_Bdr_size_;
    int         theta_Wl_size_;
    int         theta_Bl_size_;

    //Lambdas     lambdas;

  public:

    /******************************************************************************
     *                              My many friends                               *
     ******************************************************************************/

    // Needs write access to dict_
    friend RecursiveAutoencoderBase* reindex_dict(RecursiveAutoencoderBase& rae, TrainingCorpus& trainC, TrainingCorpus& testC);
    friend class Senna;

    // Is really a child class, doing the heavy lifting
    friend class SingleProp;
    friend class Backpropagator;

    // Collect gradients etc. (train_update)
    friend void setVarsAndNumber(Real *&vars, int &number_vars, Model &model);

    // Changes parameters in theta_
    friend int finite_grad_check(Model &model);

    /***************************************************************************
     *                              Serialization                              *
     ***************************************************************************/

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

}

#endif /* end of include guard: RECURSIVE_AUTOENCODER_H_JVBAVFOJ */
