#pragma once
#include <iostream>
#include <map>
#include <vector>
#include <random>

#include "ActivationFunctions.h"

class Perceptron {
public:
	Perceptron(int numInputs);

	template <typename T>
	void train(
		std::map<
		const VectorXd,
		double,
		T,
		Eigen::aligned_allocator<std::pair<const VectorXd, double> >
		> inOutPairs,
		ActivationFunctions::ActivationFunction* activationFunction,
		void (*backpropogationFunction)(const VectorXd&, VectorXd&, bool),
		double (*errorFunction)(double target, double value),
		double learningRate
	) {
		bool good = false;
		double current;
		double error;
		while(!good) {
			good = true;
			for(auto [key, value] : inOutPairs) {
				current = activationFunction->f(-key.dot(weights));
				error = errorFunction(value, current);
				if(current != value) {
					good = false;
					if(value) {
						backpropogationFunction(key, weights, true);
					} else {
						backpropogationFunction(key, weights, false);
					}
				}
			}
		}
		for(int i = 0; i < weights.size( ); i++) {
			printf("w%d %f", i, weights[i]);
		}
	};

private:
	VectorXd weights;

	std::uniform_real_distribution<double> uniformRealDistribution{ 0.0, 1.0 };
	std::default_random_engine defaultRandomEngine{ std::random_device{}() };

	double getRandomWeight( ) {
		return uniformRealDistribution(defaultRandomEngine);
	};

};