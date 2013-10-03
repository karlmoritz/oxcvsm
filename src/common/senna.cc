// File: senna.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 08-02-2013
// Last Update: Tue 01 Oct 2013 07:05:28 PM BST

/*------------------------------------------------------------------------
 * Description: <DESC> 
 * 
 *------------------------------------------------------------------------
 * History:
 * TODO:    
 *========================================================================
 */

#include <iostream>
#include <fstream>

#include <Eigen/Core>

#include "shared_defs.h"

#include "senna.h"

#include "recursive_autoencoder.h"
#include "utils.h"

using namespace std;

  Senna::Senna(RecursiveAutoencoderBase& rae, int embeddings_type)
: good_counter(0), bad_counter(0), rae(rae)
{
  if (embeddings_type >= 0)
    use_embeddings = true;
  else
    use_embeddings = false;

  /*
   * Embedding types:
   * 0 - senna
   * 1 - turian
   * 2 - cldc en
   * 3 - cldc de
   */

  if (embeddings_type == 0)
  {
    string word_file = "senna/hash/words.lst";
    string embeddings_file = "senna/embeddings/embeddings.txt";

    int count = 0;
    string line;
    {
      ifstream words_in(word_file.c_str());
      while (getline(words_in, line)) {
        words[trim(line)] = count;
        count++;
      }
    }

    Real token;
    count = 0;
    {
      ifstream embeddings_in(embeddings_file.c_str());
      while (getline(embeddings_in, line)) {
        vector<Real> instance;
        stringstream line_stream(line);
        while (line_stream >> token)
        {
          instance.push_back(token);
        }
        embeddings[count] = instance;
        count++;
      }
    }
  }
  else if (embeddings_type == 1)
  {
    // check for language!

    string embeddings_file = "../data/turian/x50.txt";
    string word;
    string line;
    Real token;
    int count = 0;
    {
      ifstream embeddings_in(embeddings_file.c_str());
      while (getline(embeddings_in, line)) {
        vector<Real> instance;
        stringstream line_stream(line);
        line_stream >> word;
        words[trim(word)] = count;
        while (line_stream >> token)
        {
          instance.push_back(token);
        }
        embeddings[count] = instance;
        count++;
      }
    }
  }
  else if (embeddings_type == 2 or embeddings_type == 3)
  {
    cout << "Reading in CLDC embeddings ";
    string embeddings_file = "";
    if (embeddings_type == 2) {
      cout << "in English" << endl;
      embeddings_file = "../data/alex/embeddings/de-en.en";
    }
    if (embeddings_type == 3) {
      cout << "in German" << endl;
      embeddings_file = "../data/alex/embeddings/de-en.de";
    }
    string word;
    string colon;
    string line;
    Real token;
    int count = 0;
    {
      ifstream embeddings_in(embeddings_file.c_str());
      while (getline(embeddings_in, line)) {
        vector<Real> instance;
        stringstream line_stream(line);
        line_stream >> word;
        line_stream >> colon;
        line_stream >> colon;
        words[trim(word)] = count;
        while (line_stream >> token)
        {
          instance.push_back(token);
        }
        embeddings[count] = instance;
        count++;
      }
    }
  }
  cout << "Read in " << words.size() << " words and " << embeddings.size() << " embeddings." << endl;
}


void Senna::applyEmbeddings()
{
  if (not use_embeddings)
    return;

  int found = 0;
  int notfound = 0;

  for (auto i = rae.dict_.min_label(); i<=rae.dict_.max_label(); ++i)
  {
    string dword = rae.dict_.label(i);
    auto j = words.find(dword);
    if (j != words.end())
    {
      found++;
      WeightVectorType x(&embeddings[j->second][0], embeddings[j->second].size());
      rae.D.row(i) = x;
    } else {
      notfound++;
    }
  }
  cout << "Found " << found << " of " << (found+notfound) << " words." << endl;

}

LabelID Senna::id(const Label& l, bool add_new)
{
  if ((not use_embeddings) or (not add_new)) {
    // return rae.dict_.id(l,add_new);
    LabelID x = rae.dict_.id(l,add_new);
    if (x != rae.dict_.m_bad_label) // Case 1: Word already in the dictionary
      ++good_counter;
    else {
      ++bad_counter;
    }
    return x;

  }
  else
  {
    // We use embeddings AND we want to add this word to the dictionary
    LabelID x = rae.dict_.id(l);
    if (x != rae.dict_.m_bad_label) // Case 1: Word already in the dictionary
      return x;
    else
    {
      auto i = words.find(l);
      if (i != words.end())           // Case 2: Word in embeddings
        return rae.dict_.id(l,true);  // add to dictionary
      else                            // Case 3: Word neither in dict nor in embeddings
        return rae.dict_.m_bad_label;
    }
  }
}
