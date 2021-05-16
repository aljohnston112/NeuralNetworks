#pragma once

#include <cmath>

#include "ComparatorVectorXd.h"

namespace ActivationFunctions {
	double sign(const VectorXd& in, const VectorXd& ws) {
		return copysign(1, in.dot(ws)) == 1 ? 1 : 0;
	};

	class ActivationFunction {
	public:
		virtual double f(double x) = 0;
		virtual double df(double x) = 0;
		virtual ~ActivationFunction( ) = 0;
	};

	ActivationFunction::~ActivationFunction( ) {

	};

	class Sigmoid: public ActivationFunction {
	public:
		double f(double x) override {
			return 1.0 / (1.0 + exp(x));
		};

		double df(double x) override {
			return f(x) * (1.0 - f(x));
		};

	};

};

namespace BackpropogationFunctions {
	void subInFromWeights(const VectorXd& in, VectorXd& ws, bool add) {
		if(add) {
			ws += in;
		} else {
			ws -= in;
		}
	};
};

namespace ErrorFunctions {
	double squared(double target, double output) {
		return pow((target - output), 2) / 2.0;
	};
}