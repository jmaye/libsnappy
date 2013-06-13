/******************************************************************************
 * Copyright (C) 2013 by Jerome Maye                                          *
 * jerome.maye@gmail.com                                                      *
 *                                                                            *
 * This program is free software; you can redistribute it and/or modify       *
 * it under the terms of the Lesser GNU General Public License as published by*
 * the Free Software Foundation; either version 3 of the License, or          *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * Lesser GNU General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the Lesser GNU General Public License   *
 * along with this program. If not, see <http://www.gnu.org/licenses/>.       *
 ******************************************************************************/

/** \file module.cpp
    \brief This file does the python bindings.
  */

#include <exception>
#include <algorithm>

#include <boost/python.hpp>
#include <boost/python/object.hpp>
#include <boost/python/handle.hpp>

#include <numpy/arrayobject.h>

#include "snappy.h"

using namespace boost::python;

/// Compresses input object
object compress(object& input) {
  if (PyArray_Check(input.ptr())) {
    PyArrayObject* a = (PyArrayObject*)PyArray_FROM_O(input.ptr());
    if (a == NULL)
      throw std::runtime_error("Could not get NP array.");
    if (a->descr->elsize != sizeof(char))
      throw std::runtime_error("Must be an 8-byte array");
    const size_t maxCompressedLength =
      snappy::MaxCompressedLength(a->dimensions[0]);
    char* compressed = reinterpret_cast<char*>(
      malloc(sizeof(char) * maxCompressedLength));
    size_t compressedLength;
    snappy::RawCompress(reinterpret_cast<const char*>(a->data),
      a->dimensions[0], compressed, &compressedLength);
    PyObject* b = PyArray_SimpleNew(a->nd,
      reinterpret_cast<npy_intp*>(&compressedLength), a->descr->type_num);
    std::copy(compressed, compressed + compressedLength,
      reinterpret_cast<char*>(reinterpret_cast<PyArrayObject*>(b)->data));
    delete compressed;
    boost::python::handle<PyObject> handle(b);
    return object(handle);
  }
  else
    throw std::runtime_error("Input is not an array");
}

/// Uncompresses input object
object uncompress(object& input) {
  if (PyArray_Check(input.ptr())) {
    PyArrayObject* a = (PyArrayObject*)PyArray_FROM_O(input.ptr());
    if (a == NULL)
      throw std::runtime_error("Could not get NP array.");
    if (a->descr->elsize != sizeof(char))
      throw std::runtime_error("Must be an 8-byte array");
    const char* compressed = reinterpret_cast<const char*>(a->data);
    size_t uncompressedLength;
    if (!snappy::GetUncompressedLength(compressed, a->dimensions[0],
        &uncompressedLength))
      throw std::runtime_error("Parsing error");
    char* uncompressed = reinterpret_cast<char*>(
      malloc(sizeof(char) * uncompressedLength));
    if (!snappy::RawUncompress(compressed, a->dimensions[0],
        uncompressed))
      throw std::runtime_error("Failed to uncompress");
    PyObject* b = PyArray_SimpleNew(a->nd,
      reinterpret_cast<npy_intp*>(&uncompressedLength), a->descr->type_num);
    std::copy(uncompressed, uncompressed + uncompressedLength,
      reinterpret_cast<char*>(reinterpret_cast<PyArrayObject*>(b)->data));
    delete uncompressed;
    boost::python::handle<PyObject> handle(b);
    return object(handle);
  }
  else
    throw std::runtime_error("Input is not an array");
}

BOOST_PYTHON_MODULE(libsnappy_python) {
  import_array();
  def("compress", &compress, "Compress an array of bytes");
  def("uncompress", &uncompress, "Uncompress an array of bytes");
}
