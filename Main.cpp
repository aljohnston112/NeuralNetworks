#pragma once

#include "Perceptron.h"

int main( ) {
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

	Perceptron p(2);
	ActivationFunctions::Sigmoid* s = new ActivationFunctions::Sigmoid( );
	double learningRate = 1.0;
	p.train<ComparatorVectorXd>(
		inOutPairs, 
		s,
		BackpropogationFunctions::subInFromWeights, 
		ErrorFunctions::squared,
		learningRate);

}