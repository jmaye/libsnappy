remake_find_package(PythonLibs)
remake_find_package(Boost COMPONENTS python RESULT_VAR Boost_FOUND)

remake_include(../core)
remake_include(${PYTHON_INCLUDE_PATH})
remake_include(${Boost_INCLUDE_DIR})

remake_add_library(snappy_python PREFIX OFF LINK snappy ${PYTHON_LIBRARIES}
  ${Boost_LIBRARIES})
