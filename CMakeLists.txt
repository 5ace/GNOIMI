project (NearestSearch CXX C)
cmake_minimum_required(VERSION 2.6)

find_package(LAPACK)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++11 -m64 -lpthread -lopenblas -lgomp -lgflags -fopenmp")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -std=c++11 -m64")

SET(YAEL_DIR ${CMAKE_SOURCE_DIR}/../yael_v438)
SET(YAEL_LIB ${CMAKE_SOURCE_DIR}/../yael_v438/yael)

SET(BLAS_DIR /usr/include/openblas)
SET(BLAS_LIB /usr/lib64)

SET(GLOG_DIR ${CMAKE_SOURCE_DIR}/../glog/src)
SET(GLOG_LIB ${CMAKE_SOURCE_DIR}/../glog/.libs)
MESSAGE( STATUS "this var key = ${GLOG_DIR}.")
#faiss @8bc09261f4f4cf66371e14f8f9cef0b4a904e84a
#LDFLAGS="-L/home/zhangcunyi/lib/mkl/lib -L/home/zhangcunyi/code/faiss" CXXFLAGS="-I/home/zhangcunyi/code" ./configure --without-cuda
SET(FAISS_DIR ${CMAKE_SOURCE_DIR}/..) #faiss @8bc09261f4f4cf66371e14f8f9cef0b4a904e84a
SET(FAISS_LIB ${CMAKE_SOURCE_DIR}/../faiss)

#########################################################################################
INCLUDE_DIRECTORIES(${BLAS_DIR})
INCLUDE_DIRECTORIES(${YAEL_DIR})
LINK_DIRECTORIES(${YAEL_LIB})
LINK_DIRECTORIES(${BLAS_LIB})
INCLUDE_DIRECTORIES(${FAISS_DIR})
LINK_DIRECTORIES(${FAISS_LIB})
INCLUDE_DIRECTORIES(${GLOG_DIR})
LINK_DIRECTORIES(${GLOG_LIB})

ADD_EXECUTABLE (learnGNOIMI learn_GNOIMI.cpp)
TARGET_LINK_LIBRARIES (learnGNOIMI yael ${LAPACK_LIBRARIES} libfaiss.a libglog.a)
#########################################################################################
ADD_EXECUTABLE (searchGNOIMI search_GNOIMI.cpp)
TARGET_LINK_LIBRARIES (searchGNOIMI yael ${LAPACK_LIBRARIES} libfaiss.a libglog.a )
#########################################################################################
ADD_EXECUTABLE (gen_init_data generate_initialization_data.cpp)
TARGET_LINK_LIBRARIES (gen_init_data yael ${LAPACK_LIBRARIES} libfaiss.a libglog.a )
#########################################################################################
ADD_EXECUTABLE (gen_groudth gen_groudth.cpp)
TARGET_LINK_LIBRARIES (gen_groudth yael ${LAPACK_LIBRARIES} libfaiss.a libglog.a )
#########################################################################################
ADD_EXECUTABLE (print_fvecs print_fvecs.cpp)
TARGET_LINK_LIBRARIES (print_fvecs yael ${LAPACK_LIBRARIES} libfaiss.a libglog.a )
#########################################################################################
ADD_EXECUTABLE (train_opq train_opq.cpp)
TARGET_LINK_LIBRARIES (train_opq yael ${LAPACK_LIBRARIES} libfaiss.a libglog.a )
