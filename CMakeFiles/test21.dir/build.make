# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/boluo/Desktop/opencv_ws/hw1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/boluo/Desktop/opencv_ws/hw1

# Include any dependencies generated for this target.
include CMakeFiles/test21.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test21.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test21.dir/flags.make

CMakeFiles/test21.dir/test21.cpp.o: CMakeFiles/test21.dir/flags.make
CMakeFiles/test21.dir/test21.cpp.o: test21.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/boluo/Desktop/opencv_ws/hw1/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/test21.dir/test21.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/test21.dir/test21.cpp.o -c /home/boluo/Desktop/opencv_ws/hw1/test21.cpp

CMakeFiles/test21.dir/test21.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test21.dir/test21.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/boluo/Desktop/opencv_ws/hw1/test21.cpp > CMakeFiles/test21.dir/test21.cpp.i

CMakeFiles/test21.dir/test21.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test21.dir/test21.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/boluo/Desktop/opencv_ws/hw1/test21.cpp -o CMakeFiles/test21.dir/test21.cpp.s

CMakeFiles/test21.dir/test21.cpp.o.requires:
.PHONY : CMakeFiles/test21.dir/test21.cpp.o.requires

CMakeFiles/test21.dir/test21.cpp.o.provides: CMakeFiles/test21.dir/test21.cpp.o.requires
	$(MAKE) -f CMakeFiles/test21.dir/build.make CMakeFiles/test21.dir/test21.cpp.o.provides.build
.PHONY : CMakeFiles/test21.dir/test21.cpp.o.provides

CMakeFiles/test21.dir/test21.cpp.o.provides.build: CMakeFiles/test21.dir/test21.cpp.o

# Object files for target test21
test21_OBJECTS = \
"CMakeFiles/test21.dir/test21.cpp.o"

# External object files for target test21
test21_EXTERNAL_OBJECTS =

test21: CMakeFiles/test21.dir/test21.cpp.o
test21: CMakeFiles/test21.dir/build.make
test21: /usr/local/lib/libopencv_calib3d.so.3.2.0
test21: /usr/local/lib/libopencv_core.so.3.2.0
test21: /usr/local/lib/libopencv_features2d.so.3.2.0
test21: /usr/local/lib/libopencv_flann.so.3.2.0
test21: /usr/local/lib/libopencv_highgui.so.3.2.0
test21: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
test21: /usr/local/lib/libopencv_imgproc.so.3.2.0
test21: /usr/local/lib/libopencv_ml.so.3.2.0
test21: /usr/local/lib/libopencv_objdetect.so.3.2.0
test21: /usr/local/lib/libopencv_photo.so.3.2.0
test21: /usr/local/lib/libopencv_shape.so.3.2.0
test21: /usr/local/lib/libopencv_stitching.so.3.2.0
test21: /usr/local/lib/libopencv_superres.so.3.2.0
test21: /usr/local/lib/libopencv_video.so.3.2.0
test21: /usr/local/lib/libopencv_videoio.so.3.2.0
test21: /usr/local/lib/libopencv_videostab.so.3.2.0
test21: /usr/local/lib/libopencv_viz.so.3.2.0
test21: /usr/local/lib/libopencv_objdetect.so.3.2.0
test21: /usr/local/lib/libopencv_calib3d.so.3.2.0
test21: /usr/local/lib/libopencv_features2d.so.3.2.0
test21: /usr/local/lib/libopencv_flann.so.3.2.0
test21: /usr/local/lib/libopencv_highgui.so.3.2.0
test21: /usr/local/lib/libopencv_ml.so.3.2.0
test21: /usr/local/lib/libopencv_photo.so.3.2.0
test21: /usr/local/lib/libopencv_video.so.3.2.0
test21: /usr/local/lib/libopencv_videoio.so.3.2.0
test21: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
test21: /usr/local/lib/libopencv_imgproc.so.3.2.0
test21: /usr/local/lib/libopencv_core.so.3.2.0
test21: CMakeFiles/test21.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable test21"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test21.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test21.dir/build: test21
.PHONY : CMakeFiles/test21.dir/build

CMakeFiles/test21.dir/requires: CMakeFiles/test21.dir/test21.cpp.o.requires
.PHONY : CMakeFiles/test21.dir/requires

CMakeFiles/test21.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test21.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test21.dir/clean

CMakeFiles/test21.dir/depend:
	cd /home/boluo/Desktop/opencv_ws/hw1 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/boluo/Desktop/opencv_ws/hw1 /home/boluo/Desktop/opencv_ws/hw1 /home/boluo/Desktop/opencv_ws/hw1 /home/boluo/Desktop/opencv_ws/hw1 /home/boluo/Desktop/opencv_ws/hw1/CMakeFiles/test21.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test21.dir/depend

