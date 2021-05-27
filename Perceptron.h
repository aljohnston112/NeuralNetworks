#pragma once
#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>
using namespace Eigen;

#include "Functions.h"
using namespace ActivationFunctions;

class Perceptron {
public:
	Perceptron(
		int numInputs,
		ActivationFunctions::ActivationFunction* activationFunction,
		ErrorFunctions::ErrorFunction* errorFunction,
		double learningRate = 1
	);

	template <typename T>
	void train(
		std::map<
		const VectorXd,
		double,
		T,
		Eigen::aligned_allocator<std::pair<const VectorXd, double>>
		> inOutPairs
	) {
		bool good = false;
		int j;
		int epochs = 0;
		while(!good) {
			epochs++;
			good = true;
			j = -1;
			for(auto [key, target] : inOutPairs) {
				j++;
				getOutput(key);
				backPropagate(target);
				if((current > (target + .001)) || (current < (target - .001))) {
					good = false;
					for(int i = 0; i < weights.size(); i++) {
						printf("w%d %f ", i, weights[i]);
					}
					printf("\n");
				}
			}
		}
		for(int i = 0; i < weights.size(); i++) {
			printf("w%d %f ", i, weights[i]);
		}
		printf("Epochs: %d", epochs);
	};

	double getOutput(const VectorXd* input) {
		currentIn = input;
		currentOut = activationFunction->f(input->dot(weights));
		return currentOut;
	};

	std::vector<double> backPropagate(double target) {
		double error;
		error = errorFunction->f(target, currentOut);
		double dEOverdOut;
		double dOutOverdNet;
		double dNetOverdW;
		double dEOverdW;
		std::vector<double> derivatives(0);
		dEOverdOut = errorFunction->df(target, currentOut);
		lastNeg[currentIn] = neg[currentIn];
		if(dEOverdOut >= 0) {
			neg[currentIn] = false;
		} else {
			neg[currentIn] = true;
		}
		if(neg[currentIn] == lastNeg[currentIn]) {
			learningRate *= 1.1;
		} else {
			learningRate *= 0.9;
		}

		dOutOverdNet = activationFunction->df(currentOut);
		for(int i = 0; i < currentIn->size(); i++) {
			dNetOverdW = (*currentIn)[i];
			dEOverdW = dEOverdOut * dOutOverdNet * dNetOverdW;
			derivatives.emplace_back(dEOverdW);
			weights[i] -= (learningRate * dEOverdW);
		}
		return derivatives;
	};

private:
	VectorXd weights;
	const VectorXd* currentIn;
	double currentOut;
	ActivationFunctions::ActivationFunction* activationFunction;
	ErrorFunctions::ErrorFunction* errorFunction;
	double learningRate;

	std::map<const VectorXd*, bool> lastNeg{};
	std::map<const VectorXd*, bool> neg{};

};