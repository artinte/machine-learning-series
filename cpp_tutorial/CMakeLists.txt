cmake_minimum_required(VERSION 3.20)

# Project name and version
project(CppTutorial VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(binary-tree-reverse binary_tree_reverse.cc)
