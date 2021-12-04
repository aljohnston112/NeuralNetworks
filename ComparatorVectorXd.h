#pragma once
#include <Eigen/Dense>

using namespace Eigen;

class ComparatorVectorXd {
public:
	bool operator()(VectorXd const& a, VectorXd const& b) const {
		return std::lexicographical_compare(
			a.data(), a.data() + a.size(),
			b.data(), b.data() + b.size());
	}
};