// File: shared_defs.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 07-01-2013
// Last Update: Thu 03 Oct 2013 11:43:24 AM BST

#ifndef SHARED_DEFS_H_FJE5IPWU
#define SHARED_DEFS_H_FJE5IPWU


// STL
#include <iostream>
#include <boost/assign.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

#include "grammarrules.h"

using namespace std;

/***************************************************************************
 *                                Typedefs                                 *
 ***************************************************************************/
#ifndef LBFGS_FLOAT
#define LBFGS_FLOAT 64
#endif

typedef double Real;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixReal;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1>              VectorReal;
typedef Eigen::Array<Real, Eigen::Dynamic, 1>               ArrayReal;

typedef Eigen::Map<MatrixReal>            WeightMatrixType;
typedef std::vector<WeightMatrixType>     WeightMatricesType;
typedef Eigen::Map<VectorReal>            WeightVectorType;
typedef Eigen::Map<ArrayReal>             WeightArrayType;
typedef std::vector<WeightVectorType>     WeightVectorsType;

typedef std::map<std::pair<int,int>, int> I2Map;
typedef std::map<std::pair<int,std::pair<int,int>>, int> I3Map;

enum TreeType {
  TREE_FOREST,
  TREE_CCG,
  TREE_STANFORD,
  TREE_ALPINO, // ~= Stanford but different format
};

enum LineSearchType { // Matching numbering from lbfgs
  MORETHUENTE = 0,
  WOLFE = 2,
  ARMIJO = 1,
  STRONGWOLFE = 3,
};

//("backtracking",WOLFE)
static std::map<string,LineSearchType> s2line_map =
boost::assign::map_list_of
("morethuente",MORETHUENTE)
("wolfe",WOLFE)
("armijo",ARMIJO)
("strongwolfe",STRONGWOLFE);

#include "dictionary.h"

struct TrainingInstance
{
  int             value;
  TrainingInstance(int val) : value(val) {}
  vector<LabelID> words;
  // Should be moved to CCGInstance. Need to solve slicing
  vector<int>     child0;
  vector<int>     child1;
  vector<int>     nodes; // -1 if treenode, >=0: pointer to word in words
  vector<CCGRuleType>     rule;
  vector<int>             cat; // = POS / phrase level tag
  vector<int>             tree_size;
};

struct ForestInstance : TrainingInstance
{
  ForestInstance(int val) : TrainingInstance(val) {}
};

struct CCGInstance : TrainingInstance
{
  /*
   * word_at_node: -1 if this is a non-leaf node, else reference the word from
   * the dict
   * child0 and child1 point to other elements in the nodes vector
   * rule describes the CCG rule applied at this node
   */

  //vector<LabelID> nodes;
  //vector<int>     child0;
  //vector<int>     child1;
  //vector<CCGRuleType>     rule;
  CCGInstance(int val) : TrainingInstance(val) {}
};

// Forward definition
class RecursiveAutoencoderBase;

typedef vector<TrainingInstance> TrainingCorpus;

template <typename T>
struct modvars
{
  T D;
  T U;
  T V;
  T W;
  T A;
  T Wd;
  T Wdr;
  T Bd;
  T Bdr;
  T Wf;
  T Wl;
  T Bl;

  T alpha_rae;
  T alpha_lbl;

  // Default constructors. Need
  modvars() { init(); }
  void init() {};
};

template<> void modvars<float>::init();
template<> void modvars<Real>::init();
template<> void modvars<int>::init();
template<> void modvars<bool>::init();

typedef modvars<float> Lambdas;
typedef modvars<bool>  Bools;
typedef modvars<int>   Counts;

struct Model
{
  TrainingCorpus corpus;
  std::vector<int> indexes;
  RecursiveAutoencoderBase& rae;
  float alpha;
  float beta;
  float gamma; // used for down-weighting nc error
  Bools bools;
  //bool calc_tree_error;
  //bool calc_label_error;

  int normalization_type;

  // Minibatch variables
  int from;
  int to;

  Model* b; // Use this if models should be trained jointly
  Model* a; // Use this if model should only use a.fProp as measure

  int it_count;

  // Really dirty data structure abusre
  int num_noise_samples;
  int noise_sample_offset;

  // L2?
  bool calc_L2;
  // bool calc_L1;

  WeightVectorsType vectorsA; // Use this if biprop should use vectorsA as target vectors
  Model(RecursiveAutoencoderBase& rae_, TrainingCorpus corp);
};

/*
 *struct BiModel
 *{
 *  Model a;
 *  Model b;
 *  TrainingCorpus& corpus;
 *
 *  // Minibatch variables
 *  int from;
 *  int to;
 *
 *  BiModel(Model& a, Model& b) : a(a), b(b), corpus(a.corpus) {}
 *};
 */

namespace SentSim
{

struct EvalInstance
{
  CCGInstance s1;
  CCGInstance s2;
  int value;
  int overlap;
  EvalInstance(CCGInstance s1, CCGInstance s2, float value)
    : s1(s1), s2(s2), value(value), overlap(0) {}
  EvalInstance(float value)
    : s1(0), s2(0), value(value), overlap(0) {}
};

typedef vector<EvalInstance> Corpus;

}

#endif /* end of include guard: SHARED_DEFS_H_FJE5IPWU */
