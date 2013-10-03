// File: reindex_dict.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 14-02-2013
// Last Update: Thu 03 Oct 2013 11:28:01 AM BST

/*------------------------------------------------------------------------
 * Description: <DESC>
 *
 *------------------------------------------------------------------------
 * History:
 * TODO:
 *========================================================================

*/

#ifndef REINDEX_DICT_H_YD7SAOWK
#define REINDEX_DICT_H_YD7SAOWK

#include "dictionary.h"
#include "shared_defs.h"

class RecursiveAutoencoderBase;

RecursiveAutoencoderBase* reindex_dict(RecursiveAutoencoderBase& rae, TrainingCorpus& trainC);

#endif /* end of include guard: REINDEX_DICT_H_YD7SAOWK */
