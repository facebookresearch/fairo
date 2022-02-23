# libfake_clock

A clock class that is convenient for testing.

This class satisfies the
[TrivialClock](http://en.cppreference.com/w/cpp/concept/TrivialClock)
requirement and as such can be used in place of any standard clock
(e.g. std::chrono::system_clock). This is useful to test C++ code that
manipulates time, without having to use sleep() in your unittests.

The clock uses an uint64_t internally, so it can store all
nanoseconds in a century. This is consistent with the precision
required of std::chrono::nanoseconds in C++11.

Example usage:

````C++
   fake_clock::time_point t1 = fake_clock::now();
   fake_clock::advance(std::chrono::milliseconds(100));
   fake_clock::time_point t2 = fake_clock::now();
   auto elapsed_us = std::chrono::duration_cast<
          std::chrono::microseconds>(t2 - t1).count();
   assert(100000 == elapsed_us);
````

For a more advanced and practical example, see
[my prometheus client implementation](https://github.com/korfuri/client_cpp/blob/c922b557ec01e9399499a05b04835cb43c2bc4c6/prometheus/client_test.cc)
for which I originally wrote this.
