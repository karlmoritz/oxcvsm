// File: train.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Thu 03 Oct 2013 11:27:36 AM BST

// STL
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// Boost
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
# if defined(RECENT_BOOST)
#include <boost/locale.hpp>
# endif

// Local
#include "common/shared_defs.h"
#include "common/dictionary.h"

#include "common/config.h"

#include "common/models.h"

#include "common/load_stanford.h"
#include "common/load_alpino.h"
#include "common/load_ccg.h"
#include "common/senna.h"
#include "common/reindex_dict.h"

#include "common/finetune_classifier.h"

// Training Regimes
#include "common/train_lbfgs.h"
#include "common/train_sgd.h"
#include "common/train_adagrad.h"

#include "common/train_update.h"
#include "common/finite_grad_check.h"

#define EIGEN_DONT_PARALLELIZE

using namespace std;
namespace bpo = boost::program_options;

int main(int argc, char **argv)
{
# if defined(RECENT_BOOST)
  cout << "Using Boost" << endl;
  boost::locale::generator gen;
  std::locale l = gen("de_DE.UTF-8");
  std::locale::global(l);
# endif

  cout << "Oxford Compositional Vector Space Modelling Package" << endl << "Copyright 2013 Karl Moritz Hermann" << endl;

  /***************************************************************************
   *                         Command line processing                         *
   ***************************************************************************/

  bpo::variables_map vm;

  // Command line processing
  bpo::options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
    ("config,c", bpo::value<string>(),
     "config file specifying additional command line options");
  bpo::options_description generic("Allowed options");
  generic.add_options()
    ("type", bpo::value<string>()->default_value("ccaeb"),
     "type of model (ccaeb, mvrnn)")

    ("input-pos", bpo::value<string>()->default_value(""),
     "corpus of positive sentences, one per line")
    ("input-neg", bpo::value<string>()->default_value(""),
     "corpus of negative sentences, one per line")
    ("extra-pos", bpo::value<string>()->default_value(""),
     "additional positive training data")
    ("extra-neg", bpo::value<string>()->default_value(""),
     "additional negative training data")

    ("model-in,m", bpo::value<string>(),
     "initial model")
    ("model-out,o", bpo::value<string>()->default_value("model"),
     "base filename of model output files")

    ("tree", bpo::value<string>()->default_value("ccg"),
     "tree type (ccg, stanford)")
    ("word-width", bpo::value<int>()->default_value(50),
     "width of word representation vectors.")
    ("iterations", bpo::value<int>()->default_value(-1),
     "(maximum) number of iterations (lbfgs default: 0 / sgd 250)")
    ("ftiterations", bpo::value<int>()->default_value(1000),
     "(maximum) number of finetune iterations")
    ("dump-frequency", bpo::value<int>()->default_value(10),
     "frequency at which to dump the model")
    ("num-sentences,n", bpo::value<int>()->default_value(0),
     "number of sentences to consider")
    ("method", bpo::value<string>()->default_value("lbfgs"),
     "training method (options: lbfgs,sgd,fgc,adagrad)")
    ("linesearch", bpo::value<string>()->default_value("armijo"),
     "LBFGS linesearch (morethuente, wolfe, armijo, strongwolfe)")

    ("embeddings", bpo::value<int>()->default_value(-1),
     "use embeddings to initialize dictionary (0=senna,1=turian,2=cldc)")

    ("batches", bpo::value<int>()->default_value(100),
     "number batches (adagrad minibatch)")
    ("ftcbatches", bpo::value<int>()->default_value(100),
     "number finetune batches (adagrad minibatch)")

    ("initI", bpo::value<bool>()->default_value(false),
     "initialize weight matrices to partial identity?")

    ("updateD", bpo::value<bool>()->default_value(true), "learn D weights?")
    ("updateF", bpo::value<bool>()->default_value(true), "learn F weights?")
    ("updateA", bpo::value<bool>()->default_value(true), "learn A weights?")
    ("updateWd", bpo::value<bool>()->default_value(true), "learn Wd weights?")
    ("updateWf", bpo::value<bool>()->default_value(true), "learn Wf weights?")
    ("updateWl", bpo::value<bool>()->default_value(true), "learn Wl weights?")

    ("calc_rae_error", bpo::value<bool>()->default_value(true),
     "consider the reconstruction error?")
    ("calc_lbl_error", bpo::value<bool>()->default_value(true),
     "consider the label error?")

    ("fonly", bpo::value<bool>()->default_value(false),
     "only finetune model (no RAE training)")
    ("f-in", bpo::value<string>(),
     "finetune model in")
    ("f-out", bpo::value<string>()->default_value("fmodel"),
     "finetune model out")

    ("lambdaD", bpo::value<float>()->default_value(0.00001), "Regularization for Embedding Vector")
    ("lambdaF", bpo::value<float>()->default_value(0.000001), "Regularization for Embedding Matrix")
    ("lambdaA", bpo::value<float>()->default_value(0.0000), "Regularization for Embedding Alpha")
    ("lambdaWd", bpo::value<float>()->default_value(0.007), "Regularization for Tree Matrices")
    ("lambdaBd", bpo::value<float>()->default_value(0.007), "Regularization for Tree Matrix Biases")
    ("lambdaWf", bpo::value<float>()->default_value(0.007), "Regularization for Matrix Composition")
    ("lambdaWl", bpo::value<float>()->default_value(0.0005), "Regularization for Label Matrices")
    ("lambdaBl", bpo::value<float>()->default_value(0.0005), "Regularization for Label Matrix Biases")

    ("lambdaFT", bpo::value<float>()->default_value(0.00001), "Regularization for Finetuning")

    ("alpha", bpo::value<float>()->default_value(0.2),
     "autoencoder error vs label error")
    ("epsilon", bpo::value<float>()->default_value(0.000001),
     "convergence parameter for LBFGS")
    ("eta", bpo::value<float>()->default_value(0.2),
     "(initial) eta for SGD")
    ("ftceta", bpo::value<float>()->default_value(0.01),
     "(initial) eta for finetune AdaGrad")

    ("norm", bpo::value<int>()->default_value(0),
     "normalization type (see train_update.cc)")
    ("dynamic-mode,d", bpo::value<int>()->default_value(0),
     "type of sentence representation: 0 (root+avg), 1 (root), 2 (avg), 3(concat-all), 4 (complicated)")
    ;
  bpo::options_description all_options;
  all_options.add(generic).add(cmdline_specific);

  store(parse_command_line(argc, argv, all_options), vm);
  if (vm.count("config") > 0) {
    ifstream config(vm["config"].as<string>().c_str());
    store(parse_config_file(config, all_options), vm);
  }
  notify(vm);

  if (vm.count("help")) {
    cout << all_options << "\n";
    return 1;
  }


  ModelData config;

  config.word_representation_size = vm["word-width"].as<int>();
  config.num_sentences = vm["num-sentences"].as<int>();

  config.model_out = vm["model-out"].as<string>();
  config.dump_freq = vm["dump-frequency"].as<int>();

  string treetype = vm["tree"].as<string>();
  if (treetype == "ccg")
    config.tree = TREE_CCG;
  if (treetype == "stanford")
    config.tree = TREE_STANFORD;
  if (treetype == "alpino")
    config.tree = TREE_ALPINO;

  if (vm["method"].as<string>() == "lbfgs")
    config.training_method = 0;
  else if (vm["method"].as<string>() == "sgd")
    config.training_method = 1;
  else if (vm["method"].as<string>() == "fgc")
    config.training_method = 2;
  else if (vm["method"].as<string>() == "adagrad")
    config.training_method = 3;
  else
    config.training_method = 0;

  Bools bools;
  bools.D   = vm["updateD"].as<bool>();
  bools.U   = vm["updateF"].as<bool>();
  bools.V   = vm["updateF"].as<bool>();
  bools.W   = vm["updateF"].as<bool>();
  bools.A   = vm["updateA"].as<bool>();
  bools.Wd  = vm["updateWd"].as<bool>();
  bools.Wdr = vm["updateWd"].as<bool>();
  bools.Bd  = vm["updateWd"].as<bool>();
  bools.Bdr = vm["updateWd"].as<bool>();
  bools.Wf  = vm["updateWf"].as<bool>();
  bools.Wl  = vm["updateWl"].as<bool>();
  bools.Bl  = vm["updateWl"].as<bool>();

  bool calc_rae_error = vm["calc_rae_error"].as<bool>();
  bool calc_lbl_error = vm["calc_lbl_error"].as<bool>();

  bool fonly      = vm["fonly"].as<bool>();

  float eta       = vm["eta"].as<float>();
  float ftceta    = vm["ftceta"].as<float>();
  float alpha     = vm["alpha"].as<float>();
  float epsilon   = vm["epsilon"].as<float>();

  Lambdas lambdas;
  lambdas.D   = vm["lambdaD"].as<float>();
  lambdas.U   = vm["lambdaF"].as<float>();
  lambdas.V   = vm["lambdaF"].as<float>();
  lambdas.W   = vm["lambdaF"].as<float>();
  lambdas.A   = vm["lambdaA"].as<float>();
  lambdas.Wd  = vm["lambdaWd"].as<float>();
  lambdas.Wdr = vm["lambdaWd"].as<float>();
  lambdas.Bd  = vm["lambdaBd"].as<float>();
  lambdas.Bdr = vm["lambdaBd"].as<float>();
  lambdas.Wf  = vm["lambdaWf"].as<float>();
  lambdas.Wl  = vm["lambdaWl"].as<float>();
  lambdas.Bl  = vm["lambdaBl"].as<float>();

  lambdas.alpha_rae = 1.0; //alpha;
  lambdas.alpha_lbl = 1.0; //(1.0 - alpha);

  float lambdaFT    = vm["lambdaFT"].as<float>();

  int dmode         = vm["dynamic-mode"].as<int>();

  int batches       = vm["batches"].as<int>();
  int ftcbatches    = vm["ftcbatches"].as<int>();

  config.init_to_I = vm["initI"].as<bool>();

  LineSearchType linesearch =s2line_map[vm["linesearch"].as<string>()];

  int iterations = vm["iterations"].as<int>();
  int ftiterations = vm["ftiterations"].as<int>();
  if (iterations == -1)
  {
    if (config.training_method == 1)
      iterations = 250;
    else if (config.training_method == 3)
      iterations = 10;
    else
      iterations = 0;
  }

/***************************************************************************
 *          Create RAE Instance and load data if model is present          *
 ***************************************************************************/

  bool create_new_dict = true;
  RecursiveAutoencoderBase* raeptr = nullptr;
  string type = vm["type"].as<string>();
  if (type == "ccaeb") {
    config.calc_rae = calc_rae_error; config.calc_lbl = calc_lbl_error;
    raeptr = new ccaeb::RecursiveAutoencoder(config,lambdas);
  } else if (type == "mvrnn") {
    config.calc_rae = false; config.calc_lbl = calc_lbl_error;
    raeptr = new mvrnn::RecursiveAutoencoder(config,lambdas);
  } else {
    cout << "Model (" << type << ") does not exist" << endl; assert(false);
  }

  RecursiveAutoencoderBase& rae = *raeptr;

  if (vm.count("model-in"))
  {
    std::ifstream ifs(vm["model-in"].as<string>());
    boost::archive::text_iarchive ia(ifs);
    ia >> rae;
    create_new_dict = false;
  }

  // Update the "history" maker in the config file
  std::stringstream ss;
  ss << rae.config.history << " | " << vm["method"].as<string>();
  if (config.training_method != 1)
    ss << "(" << vm["linesearch"].as<string>() << ")";
  ss << " it:" << iterations; // << " wcat/only:" << wcat << " " << wcatonly;
  ss << " lambdas: "  << "/" << lambdas.Wl << "/";
  ss                 << lambdaFT;
  rae.config.history = ss.str();

  /***************************************************************************
   *                   Print brief summary of model setup                    *
   ***************************************************************************/

  cerr << "################################" << endl;
  cerr << "# Config Summary" << endl;
  if (vm.count("model-in"))
    cerr << "# model-in =   " << vm["model-in"].as<string>() << endl;
  cerr << "# model-out =  " << vm["model-out"].as<string>() << endl;
  if (vm.count("f-in"))
    cerr << "# f-in =     " << vm["f-in"].as<string>() << endl;
  cerr << "# f-out =      " << vm["f-out"].as<string>() << endl;
  cerr << "# input-pos =  " << vm["input-pos"].as<string>() << endl;
  cerr << "# input-neg =  " << vm["input-neg"].as<string>() << endl;
  cerr << "# method =     " << vm["method"].as<string>() << endl;
  cerr << "# iterations = " << vm["iterations"].as<int>() << endl;
  cerr << "# word-width = " << vm["word-width"].as<int>() << endl;
  cerr << "# num-sentences = " << vm["num-sentences"].as<int>() << endl;
  cerr << "# history" << endl;
  cerr << "# " << rae.config.history << endl;
  cerr << "################################" << endl;


  /***************************************************************************
   *              Read in training data (positive and negative)              *
   ***************************************************************************/

  TrainingCorpus trainCorpus;

  string file_positive = vm["input-pos"].as<string>();
  string file_negative = vm["input-neg"].as<string>();
  string extra_positive = vm["extra-pos"].as<string>();
  string extra_negative = vm["extra-neg"].as<string>();

  int embeddings_type = vm["embeddings"].as<int>();
  {
    Senna senna(rae,embeddings_type);
    if (config.tree == TREE_CCG)
    {
      load_ccg::load(trainCorpus,file_positive,file_negative,rae,create_new_dict,senna);
      if (!extra_positive.empty())
        load_ccg::load_file(trainCorpus, extra_positive, rae, -1, false, false, 0, create_new_dict, senna);
      if (!extra_negative.empty())
        load_ccg::load_file(trainCorpus, extra_negative, rae, -1, false, false, 1, create_new_dict, senna);
    }
    else if (config.tree == TREE_STANFORD)
    {
      load_stanford::load(trainCorpus,file_positive,file_negative,rae,create_new_dict,senna);
      if (!extra_positive.empty())
        load_stanford::load_file(trainCorpus, extra_positive, rae, -1, false, false, 0, create_new_dict, senna);
      if (!extra_negative.empty())
        load_stanford::load_file(trainCorpus, extra_negative, rae, -1, false, false, 1, create_new_dict, senna);
    }
    else if (config.tree == TREE_ALPINO)
    {
      load_alpino::load(trainCorpus,file_positive,file_negative,rae,create_new_dict,senna);
      if (!extra_positive.empty())
        load_alpino::load_file(trainCorpus, extra_positive, rae, -1, false, false, 0, create_new_dict, senna);
      if (!extra_negative.empty())
        load_alpino::load_file(trainCorpus, extra_negative, rae, -1, false, false, 1, create_new_dict, senna);
      config.tree = TREE_STANFORD;
    }
    else
      assert (false);

    cout << "Dataset Size " << trainCorpus.size() << endl;

    if (create_new_dict)
    {
      // Don't initialize words randomly if we use senna embeddings
      rae.finalizeDictionary(true);
      senna.applyEmbeddings();
    }

  }



  /***************************************************************************
   *            Setup model and update dictionary if create above            *
   ***************************************************************************/


  RecursiveAutoencoderBase* rae2ptr = reindex_dict(rae,trainCorpus);
  delete raeptr;
  RecursiveAutoencoderBase& rae2 = *rae2ptr;
  Model model(rae2 ,trainCorpus); // yes, it's ugly, but the casting in lbfgs requires it

  model.bools = bools;
  model.alpha = alpha;

  model.from = 0;
  model.to = trainCorpus.size();
  if (config.num_sentences != 0)
    model.to = min(int(trainCorpus.size()),config.num_sentences);

  model.normalization_type = vm["norm"].as<int>();

  if (calc_lbl_error and file_negative.empty())
    assert(false);

  cout << "Dict size: " << rae2.getDictSize() << endl;

  /***************************************************************************
   *                              BFGS training                              *
   ***************************************************************************/

  //for (int i=0;i<1;i++)
  {
    if (not fonly)
    {
      if (config.training_method == 0)
      {
        cout << "Training with LBFGS" << endl;
        train_lbfgs(model,linesearch,iterations,epsilon);
      }
      else if (config.training_method == 1)
      {
        cout << "Training with SGD" << endl;
        train_sgd(model,iterations,eta);
      }
      else if (config.training_method == 2)
      {
        cout << "Finite Gradient Check" << endl;
        finite_grad_check(model);
      }
      else if (config.training_method == 3)
      {
        cout << "Training with AdaGrad" << endl;
        train_adagrad(model,iterations,eta,batches,lambdas.D);
      }


      /***************************************************************************
       *                 Storing model to file (default: model)                  *
       ***************************************************************************/

      {
        std::ofstream ofs(vm["model-out"].as<string>());
        boost::archive::text_oarchive oa(ofs);
        oa << model.rae;
      }
    }
    testModel(model);
    //assert(false);

    /***************************************************************************
     *                Evaluating model on test data (if given)                 *
     ***************************************************************************/

    if (!file_negative.empty()) // Finetuning only makes sense if we have labelled data
    {
      FinetuneClassifier ftc(rae2,trainCorpus,lambdaFT,alpha,dmode,ftiterations);
      ftc.eta = ftceta;
      ftc.num_batches = ftcbatches;

      if (vm.count("f-in"))
      {
        std::ifstream ifs2(vm["f-in"].as<string>());
        boost::archive::text_iarchive ia2(ifs2);
        ia2 >> ftc;
      }

      // Choose finetuning routine (should parametrize this).
      //ftc.trainLbfgs(linesearch);
      cout << "Before Ada" << endl;
      ftc.trainAdaGrad();
      cout << "Testing model ";
      testModel(model);
      cout << endl;
      ftc.evaluate();

      /***************************************************************************
       *            Store finetuned model to file (default: model_ft)            *
       ***************************************************************************/

      {
        std::ofstream ofs(vm["f-out"].as<string>());
        boost::archive::text_oarchive oa(ofs);
        oa << ftc;
      }
      delete rae2ptr;
    }
  }
}
