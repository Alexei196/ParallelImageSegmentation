# ParallelImageSegmentation
CS605 Image Segmentation

### List of dependencies:
* openCV version 4.7.0
* g++ compiler version 12.2.0
* CMake version 3.26.3

#### OpenCV Setup instructions:
* Download the source code for the above dependencies and extract the zip files to another directory.
* Install g++ compiler.
* Execute CMake and configure it:
    * Under "where is the source code", choose the path corresponding to the `opencv/sources` directory (Ex. `C:/opencv/sources`).
    * Under "where to build the binaries", choose the path corresponding to the `opencv/build` directory (Ex. `C:/opencv/build`).
    * Select "Configure". Under "Specify the generator for this project", choose the option for the g++ compiler you are using. For example, for Windows we could choose "MinGW Makefiles". Then check the box for "Specify native compilers". 
    * You will now be prompted to select the path for compiling C/C++ files. Specify the directory containing the compiler executable (ex. `C:\Mingw-w64\mingw64\bin\x86_64-w64-mingw32-g++.exe`).
    * Wait for the configuration to complete then select "Generate".
* For Windows OS, after configuring CMake, open a command prompt with admin privileges and navigate to the opencv\build directory.
    * Execute `mingw32-make` and `ming32-make install`. This may take a while.
* You will need to add the dependencies (the path leading to the `bin` folder) to your Environment Variables. For example: `C:\Program Files\CMake\bin` is a path containing the binary files for CMake and `C:\opencv\build\bin` is the one corresponding to openCV.
* NOTE: If you are using an IDE and it cannot find opencv, make sure that the `opencv\build\bin` and `opencv\build\include` paths are linked!

### Compile on command line: TODO replace with name of executable
To compile, you will need to look inside your `opencv\build\bin folder`. You will see several `.dll` files you will need to link to the compilation. Here is an example compilation line:
`g++ -o coloredRegion coloredRegion.cpp -IC:\opencv\build\include -LC:\opencv\build\lib -llibopencv_calib3d470 -llibopencv_core470 -llibopencv_dnn470 -llibopencv_features2d470 -llibopencv_flann470 -llibopencv_gapi470 -llibopencv_highgui470 -llibopencv_imgcodecs470 -llibopencv_imgproc470 -llibopencv_ml470 -llibopencv_objdetect470 -llibopencv_photo470 -llibopencv_stitching470 -llibopencv_videoio470`

### Run the program: TODO replace with name of executable
`./canny.exe`
`./kmeans.exe`
`./coloredRegion.exe`

### Cmake instructions
To start using Cmake you will have to call the 2 instructions to create the CMake build. Once that is done, you can generate the executables using the "cmake --build build" command. 
`cmake -S . -B build`
`cmake --build build`