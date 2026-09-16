# A project that consumes an installed mskip package:
#   cmake --install <mskip build dir> --prefix /some/prefix
#   cmake -S examples/consumer -B build -DCMAKE_PREFIX_PATH=/some/prefix
cmake_minimum_required(VERSION 3.14)
project(consumer LANGUAGES CXX)
find_package(mskip CONFIG REQUIRED)
add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE mskip::mskip)
