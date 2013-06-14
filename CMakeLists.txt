cmake_minimum_required(VERSION 2.6)

include(ReMake)

remake_project(
  libsnappy
  VERSION 1.1
  RELEASE beta
  SUMMARY "A fast compressor/decompressor library."
  AUTHOR "Jerome Maye"
  CONTACT "jerome.maye@mavt.ethz.ch"
  HOME "https://github.com/jmaye/libsnappy"
  LICENSE "GNU Lesser General Public License (LGPL)"
  PREFIX snappy-
  INSTALL /usr/
)

remake_add_directories(
  python
)

remake_pack_deb(
  DESCRIPTION "Main library."
)

remake_pack_deb(
  COMPONENT dev
  DESCRIPTION "Development headers."
  DEPENDS libsnappy
)

remake_pack_deb(
  COMPONENT python
  DESCRIPTION "Python bindings."
  DEPENDS libsnappy libboost-python[0-9.]* libpython2.[6-7]
)

remake_distribute_deb(
  DISTRIBUTION lucid
  SECTION libs
  UPLOAD ppa:jmaye/ethz
  DEPENDS remake doxygen pkg-config libboost-python-dev python-dev python-numpy
  PASS CMAKE_BUILD_TYPE LIBSNAPPY_GIT_REVISION
)

remake_distribute_deb(
  DISTRIBUTION precise
  SECTION libs
  UPLOAD ppa:jmaye/ethz
  DEPENDS remake doxygen pkg-config libboost-python-dev python-dev python-numpy
  PASS CMAKE_BUILD_TYPE LIBSNAPPY_GIT_REVISION
)
