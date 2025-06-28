cmake_minimum_required(VERSION 3.25)

# set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "0e5b6991-d74f-4b3d-a41c-cf096e0b2508")
# set(CMAKE_CXX_MODULE_STD ON)
set(CMAKE_CXX_STANDARD 26)
project(cpp_WSL_light LANGUAGES CXX)

# ------------------------------------------------------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU") # GCC
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
  -Wall -Wextra -Wextra-semi -Wcast-align -Wdouble-promotion -Wformat=2 \
  -Wimplicit-fallthrough -Wmisleading-indentation -Wmissing-noreturn \
  -Wnon-virtual-dtor -Wnull-dereference -Wold-style-cast -Woverloaded-virtual \
  -Wpacked -Wpedantic -Wshadow -Wno-sign-conversion -Wunused -ftemplate-backtrace-limit=0 \
	-fsanitize=address -fsanitize=undefined")

  link_libraries("-lstdc++exp")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang") # Clang
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
  -Wall -stdlib=libc++ -Wextra -Wextra-semi -Wcast-align -Wdouble-promotion -Wformat=2 \
  -Wimplicit-fallthrough -Wmisleading-indentation -Wmissing-noreturn -Wnon-virtual-dtor \
  -Wnull-dereference -Wold-style-cast -Woverloaded-virtual -Wpacked -Wpedantic -Wshadow \
  -Wno-sign-conversion -Wunused -Wunsequenced -ftemplate-backtrace-limit=0 \
  -fconstexpr-backtrace-limit=0 -fsanitize=address -fsanitize=undefined -fmodules")

  set(llvm_lib_path "/home/linuxbrew/.linuxbrew/opt/llvm/lib")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} \
    -L${llvm_lib_path} -Wl,-rpath,${llvm_lib_path}")
endif()

# ------------------------------------------------------------------------------------------------------------------------
add_executable(wNSA
  wNSA.cpp
)
