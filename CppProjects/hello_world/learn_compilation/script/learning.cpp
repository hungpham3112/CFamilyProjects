#include <iostream>
#include <ostream>


int main(int argc, char *argv[]) {
  // using std::cout, std::endl;
  // cout << "Range of integer: " << +std::numeric_limits<int>::min() 
  //   << " to " << +std::numeric_limits<int>::max() << endl;
  // cout << "Range of char: " << int((std::numeric_limits<char>::min()))
  //   << " to " << int(std::numeric_limits<char>::max()) << endl;
  // cout << "Range of float: " << std::numeric_limits<float>::min() 
  //   << " to " << std::numeric_limits<float>::max() << endl;
  // cout << "Range of double: " << std::numeric_limits<double>::min() 
  //     << " to " << std::numeric_limits<double>::max() << endl;
  
  // {float var1{12.1f};
  // double var2{12.1};

  // cout << std::fixed << std::setprecision(20) << var1 <<endl;
  // cout << std::fixed << std::setprecision(20) << var2 <<endl;
  // }
  
  // {
    
  //   typedef unsigned int UINT;
  //   UINT the_max_int{4294967295};
  //   cout << "Range of unsigned int: " << std::numeric_limits<UINT>::min() 
  //     << " to " << std::numeric_limits<UINT>::max() << endl;
  //   cout<< the_max_int <<endl;
  //   the_max_int++;
  //   cout<< the_max_int <<endl;
  //   the_max_int++;
  //   cout<< the_max_int << endl;
  // }

  // {
  //   enum Day: uint8_t{
  //     kMonday=2,
  //     kTuesday,  //3
  //     kWednesday,//4
  //     kThursday, //5
  //     kFriday,   //6
  //     kSaturday, //7
  //     kSunday    //8
  //   };

  //   Day day{kSaturday};
  //   cout << kSaturday << endl;
  //   cout << +(std::numeric_limits<uint8_t>::min()) << " to " 
  //     << +(std::numeric_limits<uint8_t>::max()) << endl;
  //   auto x{static_cast<uint8_t>(sizeof(uint8_t)) * 8};
  //   cout << "Size of day should be 1 byte (8 bits) because of uint8_t: " << x << " bits" << endl;
  //   cout << "Size of x should be 4 bytes (32 bits) because of number 8 is 4 bytes: " << sizeof(x) * 8 << " bits" << endl;
  // }
  // {
  //   int x{7};
  //   int y{5};
  //   cout << x / y << endl;
  // }
  {
    int x{5}, y{4}, temp;
    std::cout << "x " << x << "y: " << y << std::endl;
    if (x > y) {
      temp = x;
      x = y;
      y = temp;
    } else {
      temp = y;
      y = x;
      x = temp;
    }
    std::cout << "x " << x << "y: " << y << std::endl;
  }
  return 0;
}
