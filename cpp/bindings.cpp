#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "hnsw.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(hnsw_core, m) {
    nb::class_<HNSWIndex>(m, "HNSWIndex")
        .def(nb::init<size_t, size_t, size_t, uint32_t>(),
             "dim"_a, "M"_a = 16, "ef_construction"_a = 200, "seed"_a = 42)
        .def("add_batch", [](HNSWIndex& self,
                             nb::ndarray<const float, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu> vecs,
                             nb::ndarray<const uint32_t, nb::shape<-1>, nb::c_contig, nb::device::cpu> ids) {
            size_t n = vecs.shape(0);
            size_t dim = vecs.shape(1);
            const float* v_ptr = vecs.data();
            const uint32_t* id_ptr = ids.data();
            nb::gil_scoped_release release;
            for (size_t i = 0; i < n; ++i) {
                self.add_point(v_ptr + i * dim, id_ptr[i]);
            }
        }, "vecs"_a, "ids"_a)
        .def("search", [](const HNSWIndex& self,
                          nb::ndarray<const float, nb::shape<-1>, nb::c_contig, nb::device::cpu> query,
                          size_t k, size_t ef_search) {
            const float* q_ptr = query.data();
            nb::gil_scoped_release release;
            return self.search(q_ptr, k, ef_search);
        }, "query"_a, "k"_a = 10, "ef_search"_a = 64)
        .def("save", &HNSWIndex::save, "path"_a)
        .def("load", &HNSWIndex::load, "path"_a)
        .def_prop_ro("size", &HNSWIndex::size);
}