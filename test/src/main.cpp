#include <iostream>

#include "gaussgen.hpp"

int main() {
  Eigen::VectorXd x(5);
  x << 1., 2., 3., 4., 5.;
  Eigen::MatrixXd cov(5, 5);
  cov << 0.1, 0.01, -0.01, 0.001, 0.001, 0.01, 0.2, 0.03, -0.02, -0.03, -0.01,
      0.03, 0.3, 0.001, 0.00, 0.001, -0.02, 0.001, 0.4, -0.02, 0.001, -0.03,
      0.00, -0.02, 0.5;
  TRandom rnd;
  Eigen::VectorXd result = gaussgen::gaussgen(x, cov, &rnd);
  std::cout << result << std::endl;
  return 0;
}
