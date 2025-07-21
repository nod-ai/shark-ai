// RUN: %test_exe | FileCheck %s

#include <iostream>

int main() {
  // CHECK: Hello sharkfuser!
  std::cout << "Hello sharkfuser!" << std::endl;
  return 0;
}
