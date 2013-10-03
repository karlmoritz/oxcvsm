// File: load_ccg.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 22-01-2013
// Last Update: Thu 03 Oct 2013 11:23:10 AM BST

// STL
#include <iostream>
#include <fstream>
#include <queue>

#include "load_ccg.h"
#include "dictionary.h"

using namespace pugi;
using namespace std;

struct qelem
{
  xml_node node;
  int id;
  qelem(xml_node n, int i) : node(n), id(i) {}
};

typedef queue<qelem> xqueue;

void load_ccg::load(TrainingCorpus& trainCorpus, string file_positive, string
    file_negative, RecursiveAutoencoderBase& rae, bool add_to_dict, Senna&
    senna)
{
  load_file(trainCorpus,file_positive,rae,-1,false,false,0,add_to_dict,senna);
  if (!file_negative.empty()) // rae only training doesn't have a negative file ...
    load_file(trainCorpus,file_negative,rae,-1,false,false,1,add_to_dict,senna);

  cout << "Train Set Size " << trainCorpus.size() << endl;
}

void load_ccg::load_file(TrainingCorpus& corpus, string file_name, RecursiveAutoencoderBase&
    rae, int cv_split, bool use_cv, bool load_test, int label, bool add_to_dict, Senna& senna)
{
  int counter = 0;
  string line, token;

  pugi::xml_document doc;
  if (!doc.load_file(file_name.c_str())) assert(false);
  xml_node root = doc.child("candc");
  for (xml_node sentence = root.first_child(); sentence; sentence = sentence.next_sibling())
  {
    if (use_cv)
    {
      if (load_test && counter%10==cv_split)
      {
        CCGInstance instance(label);
        createCCGInstance(instance, sentence, rae, add_to_dict, senna);
        corpus.push_back(instance);
      }
      else if (not load_test && counter%10 != cv_split)
      {
        CCGInstance instance(label);
        createCCGInstance(instance, sentence, rae, add_to_dict, senna);
        corpus.push_back(instance);
      }
    }
    else
    {
      CCGInstance instance(label);
      createCCGInstance(instance, sentence, rae, add_to_dict, senna);
      corpus.push_back(instance);
    }
    counter++;
  }
}

void load_ccg::createCCGInstance(CCGInstance& instance, xml_node& sentence, RecursiveAutoencoderBase& rae, bool add_to_dict, Senna& senna)
{

  xqueue q;

  // Variables
  std::string parent_name;
  std::string field_;
  int cat_;
  CCGRuleType rule_;

  int counter = 0;
  xml_node active_child;
  q.push(qelem(sentence.first_child(),counter));
  counter ++;

  while (not q.empty())
  {
    qelem parent = q.front();
    q.pop();
    parent_name = parent.node.name();
    instance.child0.push_back(-1);
    instance.child1.push_back(-1);

    if(parent_name == "rule")
    {
      field_ = parent.node.attribute("type").value();
      rule_ = (s2r_map.find( field_ ) != s2r_map.end()) ? s2r_map[field_] : OTHER;
      field_ = parent.node.attribute("cat").value();
      cat_ = (c2r_map.find( field_ ) != c2r_map.end()) ? c2r_map[field_] : 0;
      //cout << "Cat " << field_ << ": " << cat_ << endl;
      assert(rule_ != LEAF);

      instance.nodes.push_back(-1);
      instance.rule.push_back(rule_);
      instance.cat.push_back(cat_);
      //cout << "Rule: " << field_ << " " << rule_ << endl;
    }
    else if(parent_name == "lf")
    {
      field_ = parent.node.attribute("cat").value();
      cat_ = (c2r_map.find( field_ ) != c2r_map.end()) ? c2r_map[field_] : 0;
      field_ = parent.node.attribute("word").value();
      instance.words.push_back(senna.id(field_,add_to_dict));
      instance.nodes.push_back(instance.words.size()-1);
      instance.rule.push_back(LEAF);
      instance.cat.push_back(cat_);
    }
    else
    {
      cout << "PN" << parent_name << endl;
      assert(false);
    }

    xml_node active_child = parent.node.first_child();
    if (active_child)
    {
      q.push(qelem(active_child,counter));
      instance.child0[parent.id] = counter;
      counter++;

      active_child = active_child.next_sibling();
      if (active_child)
      {
        q.push(qelem(active_child,counter));
        instance.child1[parent.id] = counter;
        counter++;

        assert (not active_child.next_sibling());
      }
    }
  }
}
