// File: shared_defs.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 09-05-2013
// Last Update: Thu 03 Oct 2013 11:43:12 AM BST

#include "shared_defs.h"

template<> void modvars<float>::init() { D = 0.0; U = 0.0; V = 0.0; W = 0.0; A = 0.0; Wd = 0.0; Wdr = 0.0; Bd = 0.0; Bdr = 0.0; Wf = 0.0; Wl = 0.0; Bl = 0.0; alpha_rae = 0.0; alpha_lbl = 0.0; }
template<> void modvars<Real>::init() { D = 0.0; U = 0.0; V = 0.0; W = 0.0; A = 0.0; Wd = 0.0; Wdr = 0.0; Bd = 0.0; Bdr = 0.0; Wf = 0.0; Wl = 0.0; Bl = 0.0; alpha_rae = 0.0; alpha_lbl = 0.0; }
template<> void modvars<int>::init() { D = 0; U = 0; V = 0; W = 0; A = 0; Wd = 0; Wdr = 0; Bd = 0; Bdr = 0; Wf = 0; Wl = 0; Bl = 0; alpha_rae = 0; alpha_lbl = 0; }
template<> void modvars<bool>::init() { D = true; U = true; V = true; W = true; A = true; Wd = true; Wdr = true; Bd = true; Bdr = true; Wf = true; Wl = true; Bl = true; alpha_rae = true; alpha_lbl = true; }

Model::Model(RecursiveAutoencoderBase& rae_, TrainingCorpus corp) :
  corpus(corp), rae(rae_), alpha(0.2), beta(0.5), gamma(0.1),
  normalization_type(0), b(nullptr), a(nullptr), it_count(0),
  num_noise_samples(2), noise_sample_offset(1), calc_L2(true) {

    indexes.reserve(corp.size());
    for (int i = 0; i < corp.size(); ++i)
      indexes.push_back(i);

  }
