// File: load_alpino.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Thu 03 Oct 2013 11:22:56 AM BST

#ifndef LOAD_ALPINO_H_5V0TLQNF
#define LOAD_ALPINO_H_5V0TLQNF

#include "../pugi/pugixml.hpp"

#include "shared_defs.h"
#include "recursive_autoencoder.h"
#include "senna.h"

using namespace pugi;

namespace load_alpino {
void load(TrainingCorpus& trainCorpus, string file_positive, string
    file_negative, RecursiveAutoencoderBase& rae, bool add_to_dict, Senna&
    senna);

void load_file(TrainingCorpus& corpus, string file_name, RecursiveAutoencoderBase&
    rae, int cv_split, bool use_cv, bool load_test, int label, bool add_to_dict,
    Senna& senna);

void createCCGInstance(CCGInstance& instance, xml_node& sentence,
    RecursiveAutoencoderBase& rae, bool add_to_dict, Senna& senna);
};

#endif /* end of include guard: LOAD_ALPINO_H_5V0TLQNF */
