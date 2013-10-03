// File: utils.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 30-01-2013
// Last Update: Thu 03 Oct 2013 11:47:47 AM BST

// STL
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


// Boost
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

// Local
#include "utils.h"
#include "recursive_autoencoder.h"
#include "singlepropbase.h"
#include "backpropagatorbase.h"
#include "recursive_autoencoder.h"

void dumpModel(Model& model, int k)
{
  {
  std::stringstream fname;
  k = k + model.rae.config.cycles_so_far;
  fname << model.rae.config.model_out << "_i" << k;
  std::ofstream ofs(fname.str());
  boost::archive::text_oarchive oa(ofs);
  oa << model.rae;
  }
}

void printSentence(const Dictionary& dict, const TrainingInstance &sent) {
  for (auto i = 0; i < sent.words.size() ; ++i) {
    cout << dict.label(sent.words[i]) << " ";
  }
  cout << endl;
}
