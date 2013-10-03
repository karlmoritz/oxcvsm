// File: reindex_dict.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 14-02-2013
// Last Update: Thu 03 Oct 2013 11:30:14 AM BST

#include "reindex_dict.h"
#include "recursive_autoencoder.h"

/***************************************************************************
 *           Take an RAE and reindex it based on a given corpus            *
 ***************************************************************************/
RecursiveAutoencoderBase* reindex_dict(RecursiveAutoencoderBase& rae, TrainingCorpus& trainC)
{
  RecursiveAutoencoderBase* new_rae = rae.cloneEmpty();
  std::map<LabelID,LabelID> n2o_map;

  // A: Populate new dictionary based on old one
  for (auto instance = trainC.begin(); instance != trainC.end(); ++instance)
    for (auto word = instance->words.begin() ;
        word != instance->words.end() ;
        ++word)
    {
      n2o_map[new_rae->dict_.id(rae.dict_.label(*word),true)] = *word;
      *word = new_rae->dict_.id(rae.dict_.label(*word),true);
    }

  new_rae->init(false,true);
  new_rae->initFromWithDict(rae,n2o_map);
  cout << "Reindexed dictionary from " << rae.getDictSize();
  cout << " entries down to " << new_rae->getDictSize() << "." << endl;
  return new_rae;
}
