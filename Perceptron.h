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
		double dEOverdOut;
		std::vector<bool> lastNeg(inOutPairs.size(), true);
		std::vector<bool> neg(inOutPairs.size(), true);
		double dOutOverdNet;
		double dNetOverdW;
		double dW;
		double lr = learningRate;
		int j;
		int epochs = 0;
		while (!good) {
			epochs++;
			good = true;
			j = -1;
			for (auto [key, target] : inOutPairs) {
				j++;
				current = activationFunction->f(key.dot(weights));
				if (((current > (target + .001)) || (current < (target - .001))) && lr > 0.001) {
					good = false;
					error = errorFunction(target, current);
					dEOverdOut = -(target - current);
					lastNeg[j] = neg[j];
					if (dEOverdOut >= 0) {
						neg[j] = false;
					}
					else {
						neg[j] = true;
					}
					dOutOverdNet = activationFunction->df(current);
					for (int i = 0; i < key.size(); i++) {
						dNetOverdW = key[i];
						dW = dEOverdOut * dOutOverdNet * dNetOverdW;
						weights[i] -= (lr * dW);
					}
					if (neg[j] != lastNeg[j]) {
						lr *= 0.9;
					}
					else {
						lr *= 1.1;
					}
					for (int i = 0; i < weights.size(); i++) {
						printf("w%d %f ", i, weights[i]);
					}
					printf("\n");
				}
			}
		}
		for (int i = 0; i < weights.size(); i++) {
			printf("w%d %f ", i, weights[i]);
		}
		printf("Epochs: %d", epochs);
	};

private:
	VectorXd weights;

};