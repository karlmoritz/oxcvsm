// File: fast_math.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 23-01-2013
// Last Update: Tue 01 Oct 2013 07:09:24 PM BST

/*------------------------------------------------------------------------
 * Description: <DESC> 
 * 
 *------------------------------------------------------------------------
 * History:
 * TODO:    
 *========================================================================
 */

#include <vector>
 
#include "shared_defs.h"

Real getSigmoid(Real val);
Real getExp(Real val);
Real getTanh(Real val);
Real getArgtanh(Real val);

Real getSpearmansRho(std::vector<float> s1, std::vector<float> s2);
std::vector<float> getSplitRank(std::vector<float>& v);

MatrixReal tanh_p(VectorReal& x); // takes unnormalized tan vector

Real rectifiedLinear(Real val);
Real rectifiedLinearGrad(Real val);
   
template <typename T> inline constexpr
int signum(T x, std::false_type is_signed) {
      return T(0) < x;
}

template <typename T> inline constexpr
int signum(T x, std::true_type is_signed) {
      return (T(0) < x) - (x < T(0));
}

template <typename T> inline constexpr
int signum(T x) {
      return signum(x, std::is_signed<T>());
}
