include_directories(${CMAKE_SOURCE_DIR})
include_directories(${openfst_SOURCE_DIR}/src/include)

pybind11_add_module(_kaldifst
  arc.cc
  const-fst.cc
  determinize-star.cc
  expanded-fst.cc
  float-weight.cc
  fst.cc
  fstarcsort.cc
  fstcompile.cc
  fstdeterminize.cc
  fstdraw.cc
  fstext-utils.cc
  fstinfo.cc
  fstinvert.cc
  fstminimize.cc
  fstreverse.cc
  fstrmepsilon.cc
  kaldifst.cc
  mutable-fst.cc
  symbol-table.cc
  table-matcher.cc
  vector-fst.cc
)
target_link_libraries(_kaldifst PRIVATE kaldifst_core fstscript)
if(UNIX AND NOT APPLE)
  target_link_libraries(_kaldifst PUBLIC ${PYTHON_LIBRARY})
endif()
