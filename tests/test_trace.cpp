#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

#define t_edge(...) \
    { {__VA_ARGS__}, true }
#define f_edge(...) \
    { {__VA_ARGS__}, false }

template<
    typename T,
    typename N = std::unordered_set<std::pair<std::string, std::string>>,
    typename F = std::unordered_map<std::string, std::pair<std::string, std::string>>>
auto trace_two(const T& tensor, const N& pairs, const F& fuses = {}) {
    auto traced_tensor_0 = tensor.trace(pairs, fuses);
    N double_names;
    std::vector<std::string> names;
    std::vector<typename T::edge_t> edges;
    for (const auto& [n0, n1] : pairs) {
        double_names.insert({n0, n0});
        double_names.insert({n1, n1});
        names.push_back(n0);
        names.push_back(n1);
        edges.push_back(tensor.edges(n0).conjugate());
        edges.push_back(tensor.edges(n1).conjugate());
    }
    auto identity = T(names, edges).identity_(pairs);
    if (fuses.size() != 0) {
        // identity *= tee
        for (const auto& [out, ins] : fuses) {
            const auto& [in_0, in_1] = ins;
            auto dimension = tensor.edges(in_0).total_dimension();
            auto tee = T({out, in_0, in_1}, {dimension, dimension, dimension}).zero_();
            for (TAT::Size i = 0; i < dimension; i++) {
                tee.at(std::vector<TAT::Size>{i, i, i}) = 1;
            }
            identity = identity.contract(tee, {});
            double_names.insert({in_0, in_0});
            double_names.insert({in_1, in_1});
        }
    }
    auto traced_tensor_1 = tensor.contract(identity, double_names);
    return std::make_pair(traced_tensor_0, traced_tensor_1);
}

template<typename P>
auto diff_of_pair(const P& p) {
    return (p.first - p.second).template norm<-1>();
}

#define CHECK_PAIR(x) ASSERT_FLOAT_EQ(diff_of_pair(x), 0)

TEST(test_trace, no_symmetry) {
    CHECK_PAIR(trace_two(TAT::Tensor<double, TAT::NoSymmetry>({"A", "B", "C", "D", "E"}, {2, 3, 2, 3, 4}).range_(), {{"A", "C"}, {"B", "D"}}));
    CHECK_PAIR(trace_two(TAT::Tensor<double, TAT::NoSymmetry>({"A", "B", "C"}, {2, 2, 3}).range_(), {{"A", "B"}}));
    auto a = TAT::Tensor<double, TAT::NoSymmetry>({"A", "B", "C"}, {4, 3, 5}).range_();
    auto b = TAT::Tensor<double, TAT::NoSymmetry>({"D", "E", "F"}, {5, 4, 6}).range_();
    CHECK_PAIR(trace_two(a.contract(b, {}), {{"A", "E"}, {"C", "D"}}));
}

TEST(test_trace, u1_symmetry) {
    auto a = (TAT::Tensor<double, TAT::U1Symmetry>{
        {"A", "B", "C", "D"},
        {t_edge({-1, 1}, {0, 1}, {-2, 1}), f_edge({0, 1}, {1, 2}), f_edge({0, 2}, {1, 2}), t_edge({0, 2}, {-1, 1}, {-2, 2})}}
                  .range_());
    auto b = (TAT::Tensor<double, TAT::U1Symmetry>{
        {"E", "F", "G", "H"},
        {f_edge({0, 2}, {1, 1}), t_edge({-2, 1}, {-1, 1}, {0, 2}), t_edge({0, 1}, {-1, 2}), f_edge({0, 2}, {1, 1}, {2, 2})}}
                  .range_());
    auto c = a.contract(b, {});
    auto d = trace_two(c, {{"B", "G"}});
    CHECK_PAIR(d);
    auto e = trace_two(d.first, {{"H", "D"}});
    CHECK_PAIR(e);
    auto f = trace_two(c, {{"G", "B"}, {"D", "H"}});
    CHECK_PAIR(f);
    ASSERT_FLOAT_EQ((e.first - f.first).norm<-1>(), 0);
}

TEST(test_trace, fermi_symmetry) {
    auto a = (TAT::Tensor<double, TAT::FermiSymmetry>{
        {"A", "B", "C", "D"},
        {t_edge({-1, 1}, {0, 1}, {-2, 1}), f_edge({0, 1}, {1, 2}), f_edge({0, 2}, {1, 2}), t_edge({-2, 2}, {-1, 1}, {0, 2})}}
                  .range_());
    auto b = (TAT::Tensor<double, TAT::FermiSymmetry>{
        {"E", "F", "G", "H"},
        {f_edge({0, 2}, {1, 1}), t_edge({-2, 1}, {-1, 1}, {0, 2}), t_edge({0, 1}, {-1, 2}), f_edge({2, 2}, {1, 1}, {0, 2})}}
                  .range_());
    auto c = a.contract(b, {});
    auto d = trace_two(c, {{"B", "G"}});
    CHECK_PAIR(d);
    auto e = trace_two(d.first, {{"H", "D"}});
    CHECK_PAIR(e);
    auto f = trace_two(c, {{"G", "B"}, {"D", "H"}});
    CHECK_PAIR(f);
    ASSERT_FLOAT_EQ((e.first - f.first).norm<-1>(), 0);
}

TEST(test_trace, fuse) {
    auto a = (TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B", "C", "D"}, {4, 4, 4, 4}}.range_());
    auto b = (TAT::Tensor<double, TAT::NoSymmetry>{{"E", "F", "G", "H"}, {4, 4, 4, 4}}.range_());
    auto c = a.contract(b, {});
    auto d = trace_two(c, {{"B", "G"}}, {{"X", {"C", "F"}}});
    CHECK_PAIR(d);
    auto e = trace_two(d.first, {}, {{"Y", {"A", "H"}}});
    CHECK_PAIR(e);
    auto f = trace_two(c, {{"G", "B"}}, {{"X", {"F", "C"}}, {"Y", {"A", "H"}}});
    CHECK_PAIR(f);
    ASSERT_FLOAT_EQ((e.first - f.first).norm<-1>(), 0);
}
