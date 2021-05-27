#pragma once

#include "Perceptron.h"
#include "FeedForward.h"

int main() {
	VectorXd v(3);
	v << 1, 2, 1;
	VectorXd v1(3);
	v1 << 2, 3, 1;
	VectorXd v2(3);
	v2 << 2, -1, 1;
	VectorXd v3(3);
	v3 << 3, -1, 1;

	std::map<
		const VectorXd,
		double,
		ComparatorVectorXd,
		Eigen::aligned_allocator<std::pair<const VectorXd, double>>
	> inOutPairs{
					{v, 1},
					{v1, 1},
					{v2, 0},
					{v3, 0}
	};

	Perceptron p(2, new ActivationFunctions::Sigmoid, new ErrorFunctions::Squared);
	ActivationFunctions::Sigmoid* s = new ActivationFunctions::Sigmoid();
	double learningRate = 1.0;
	p.train<ComparatorVectorXd>(inOutPairs);

	FeedForward ff(2, 0, std::vector<int>(0), 1);

	VectorXd v4(3);
	v << .05, .01, 1;
	VectorXd v5(3);
	v1 << .1, .99, 1;

	std::map<
		const VectorXd,
		double,
		ComparatorVectorXd,
		Eigen::aligned_allocator<std::pair<const VectorXd, double>>
	> inOutPairs2{
					{v4, 1},
					{v5, 1},
	};
}