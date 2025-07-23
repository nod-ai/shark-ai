// RUN: %test_exe | iree-opt --verify-roundtrip
// RUN: %test_exe | filecheck %s

#include <iostream>

int main() {
  // CHECK: func.func @main()
  std::cout << "module {" << std::endl;
  std::cout << "  func.func @main() -> i32 {" << std::endl;
  std::cout << "    %c0 = arith.constant 0 : i32" << std::endl;
  std::cout << "    return %c0 : i32" << std::endl;
  std::cout << "  }" << std::endl;
  std::cout << "}" << std::endl;
  return 0;
}
