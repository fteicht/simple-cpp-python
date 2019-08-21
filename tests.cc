#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <cmath>
#include "code.hh"

#ifdef HAVE_BACKWARD
#include "backward.hpp"

namespace backward {

backward::SignalHandling sh;

} // namespace backward
#endif

TEST_CASE("Test the approximate computation of PI/8.0", "[PI computation]") {
    double r1 = cpp_inner_loop(10);
    double r2 = cpp_inner_loop(100);
    double r3 = cpp_inner_loop(1000);
    double v = std::acos(-1.0) / 8.0;
    REQUIRE( std::fabs(v - r1) < 1e-1 );
    REQUIRE( std::fabs(v - r2) < 1e-3 );
    REQUIRE( std::fabs(v - r3) < 1e-5 );
}

// Run the following line only to test backward, it may hang your program!
// TEST_CASE("Stupid test to demonstrate Backward", "[backward]") {
//     int* v = nullptr;  // v is a null pointer
//     REQUIRE( v[0] == 0 );  // crash: Backward provides a stack trace 
// }
