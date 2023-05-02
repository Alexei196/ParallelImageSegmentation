function(find_static_library LIB_NAME OUT)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")    
    unset(FOUND CACHE)
    find_library(
        FOUND_${LIB_NAME}_STATIC
        ${LIB_NAME}
        )

    if (FOUND_${LIB_NAME}_STATIC)
        get_filename_component(ABS_FILE ${FOUND_${LIB_NAME}_STATIC} ABSOLUTE)
    else()
        message(SEND_ERROR "Unable to find library ${LIB_NAME}")
    endif()

    set(${OUT} ${ABS_FILE} PARENT_SCOPE)

endfunction()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
set(BUILD_SHARED_LIBS off)
set(OpenCV_STATIC ON)
include_directories( ${OpenCV_INCLUDE_DIRS})
add_executable( serialkmeans imageSegmentation.cpp)
set_target_properties(serialkmeans PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/executables)
find_static_library( ${OpenCV_LIBS} CV_STATIC_LIBS)
target_link_libraries( serialkmeans PUBLIC ${OpenCV_LIBS} "-static" OpenMP::OpenMP_CXX)