cmake_minimum_required(VERSION 3.20)
set(CMAKE_INSTALL_PREFIX /home/605/rossma/KmeansProject/opencv_install)
set(CMAKE_PREFIX_PATH /usr/lib64/openmpi)
project(kmeans_project)

add_subdirectory(Code/MPI)
add_subdirectory(Code/Serial)
add_subdirectory(Code/UPCXX)