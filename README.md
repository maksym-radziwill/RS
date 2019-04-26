# RS

Implementation of the Riemann-Siegel formula on the GPU + MPI

=== COMPILING

On an already configured system it is enough to run

   cd build; cmake ../CMake; make
   
The program is then compiled in the file rs in the build directory

If cmake fails because arb or flint are installed in an unusual directory
then this directory should be passed to CMAKE through the variables
CMAKE_INCLUDE_PATH and CMAKE_LIBRARY_PATH

=== RUNNING ON EC2

On an unconfigured Amazon ec2 system (either p2,p3,g2 or g3) one can run
the script ./setup_ec2.sh to quickly configure the system

