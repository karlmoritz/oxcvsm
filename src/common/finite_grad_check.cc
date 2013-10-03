// File: finite_grad_check.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 01-01-2013
// Last Update: Thu 03 Oct 2013 11:44:36 AM BST

// STL
#include <iostream>
#include <cmath>

// Boost
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

#include "shared_defs.h"

// L-BFGS
#include <lbfgs.h>

// Local
#include "finite_grad_check.h"
#include "train_update.h"
#include "recursive_autoencoder.h"

using namespace std;
namespace bpo = boost::program_options;


int finite_grad_check(Model &model)
{
  Real* vars = nullptr;
  int number_vars = 0;

  modvars<int> counts;
  model.rae.setIncrementalCounts(counts,vars,number_vars);

  WeightVectorType theta(vars,number_vars);

  Real* data1 = new Real[number_vars]();
  Real* data2 = new Real[number_vars]();
  WeightVectorType grad(data1,number_vars);
  WeightVectorType grad2(data2,number_vars);

  Real error1 = computeCostAndGrad(model,nullptr,data1,number_vars);

  modvars<Real> dists;
  Real dist = 0.0;

  Real delta = 1.0e-7;

  for (int i=0;i<number_vars;++i)
  {
    theta[i] += delta;
    Real error2 = computeCostAndGrad(model,nullptr,data2,number_vars);
    Real xdev = (error2 - error1) / delta;
    theta[i] -= delta;
    if (i < counts.D)   { dists.D += abs(grad2[i] - xdev);   cout << "D   "; }
    else if (i < counts.U)  { dists.U += abs(grad2[i] - xdev);  cout << "U  "; }
    else if (i < counts.V)  { dists.V += abs(grad2[i] - xdev);  cout << "V  "; }
    else if (i < counts.W)  { dists.W += abs(grad2[i] - xdev);  cout << "W  "; }
    else if (i < counts.A)  { dists.A += abs(grad2[i] - xdev);  cout << "A  "; }
    else if (i < counts.Wd)  { dists.Wd += abs(grad2[i] - xdev);  cout << "Wd  "; }
    else if (i < counts.Wdr) { dists.Wdr += abs(grad2[i] - xdev); cout << "Wdr "; }
    else if (i < counts.Bd)  { dists.Bd += abs(grad2[i] - xdev);  cout << "Bd  "; }
    else if (i < counts.Bdr) { dists.Bdr += abs(grad2[i] - xdev); cout << "Bdr "; }
    else if (i < counts.Wf)  { dists.Wf += abs(grad2[i] - xdev);  cout << "Wf  "; }
    else if (i < counts.Wl)  { dists.Wl += abs(grad2[i] - xdev);  cout << "Wl  "; }
    else if (i < counts.Bl)  { dists.Bl += abs(grad2[i] - xdev);  cout << "Bl  "; }

    //template<> void modvars<int>::init() { D = 0; U = 0; V = 0; W = 0; A = 0; Wd = 0; Wdr = 0; Bd = 0; Bdr = 0; Wf = 0; Wl = 0; Bl = 0; alpha_rae = 0; alpha_lbl = 0; }
    cout << i << ": " << grad2[i] << " vs " << (xdev) << "   " << error2 << " - " << error1 << "[" << theta[i] << "]" << endl;

    dist += abs(grad2[i] - xdev);
  }

  cout << "total: " << dist << " D/U/V/W/A/Wd/Wdr/Bd/Bdr/Wf/Wl/Bl " << endl;
  cout << dists.D << " ";
  cout << dists.U << " ";
  cout << dists.V << " ";
  cout << dists.W << " ";
  cout << dists.A << " ";
  cout << dists.Wd << " " << dists.Wdr << " " << dists.Bd << " " << dists.Bdr << " ";
  cout << dists.Wf << " ";
  cout << dists.Wl << " " << dists.Bl << endl;
  // Don't care about what's next
  assert(false);

  return 0;
}
