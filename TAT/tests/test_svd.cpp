#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

#define t_edge(...) \
    { {__VA_ARGS__}, true }
#define f_edge(...) \
    { {__VA_ARGS__}, false }

template<typename T>
auto check_unitary(const T& tensor, const std::string& name, const std::string& name_prime) {
    auto pairs = std::unordered_set<std::pair<std::string, std::string>>();
    for (auto n : tensor.names()) {
        if (n != name) {
            pairs.insert({n, n});
        }
    }
    auto conjugated = tensor.conjugate(true).edge_rename({{name, name_prime}});
    auto product = tensor.contract(conjugated, pairs);
    auto identity = product.same_shape().identity_({{name, name_prime}});
    if constexpr (T::symmetry_t::is_fermi_symmetry) {
        // some sign maybe wrong
        product.transform_([](auto x) { return std::abs(x); });
        identity.transform_([](auto x) { return std::abs(x); });
    }
    return (product - identity).template norm<-1>();
}

TEST(test_svd, no_symmetry) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B", "C", "D"}, {2, 3, 4, 5}}.range_();
    auto [u, s, v] = a.svd({"C", "A"}, "E", "F", "U", "V");
    ASSERT_NEAR(check_unitary(u, "E", "E'"), 0, 1e-8);
    ASSERT_NEAR(check_unitary(v, "F", "F'"), 0, 1e-8);
    auto b = v.contract(s, {{"F", "V"}}).contract(u, {{"U", "E"}});
    ASSERT_NEAR((a - b).norm<-1>(), 0, 1e-8);
}

TEST(test_svd, no_symmetry_cut) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B", "C", "D"}, {5, 4, 3, 2}}.range_();
    auto [u, s, v] = a.svd({"B", "D"}, "E", "F", "U", "V", 2ul);
    ASSERT_NEAR(check_unitary(u, "E", "E'"), 0, 1e-8);
    ASSERT_NEAR(check_unitary(v, "F", "F'"), 0, 1e-8);
    auto b = v.contract(s, {{"F", "V"}}).contract(u, {{"U", "E"}});
    ASSERT_NEAR((a - b).norm<-1>(), 0, 1e-8);
}

TEST(test_svd, u1_symmetry) {
    auto a = (TAT::Tensor<double, TAT::U1Symmetry>{
        {"A", "B", "C", "D"},
        {t_edge({-1, 1}, {0, 1}, {-2, 1}), f_edge({0, 1}, {1, 2}), f_edge({0, 2}, {1, 2}), t_edge({-2, 2}, {-1, 1}, {0, 2})}}
                  .range_());
    auto [u, s, v] = a.svd({"B", "D"}, "E", "F", "U", "V");
    ASSERT_NEAR(check_unitary(u, "E", "E'"), 0, 1e-8);
    ASSERT_NEAR(check_unitary(v, "F", "F'"), 0, 1e-8);
    auto b = v.contract(s, {{"F", "V"}}).contract(u, {{"U", "E"}});
    ASSERT_NEAR((a - b).norm<-1>(), 0, 1e-8);
}

TEST(test_svd, u1_symmetry_cut) {
    auto a = (TAT::Tensor<double, TAT::U1Symmetry>{
        {"A", "B", "C", "D"},
        {t_edge({-1, 1}, {0, 1}, {-2, 1}), f_edge({0, 1}, {1, 2}), f_edge({0, 2}, {1, 2}), t_edge({-2, 2}, {-1, 1}, {0, 2})}}
                  .range_());
    auto [u, s, v] = a.svd({"C", "A"}, "E", "F", "U", "V", 7ul);
    ASSERT_NEAR(check_unitary(u, "E", "E'"), 0, 1e-8);
    ASSERT_NEAR(check_unitary(v, "F", "F'"), 0, 1e-8);
    auto b = v.contract(s, {{"F", "V"}}).contract(u, {{"U", "E"}});
    ASSERT_NEAR((a - b).norm<-1>(), 0, 1e-8);
}

TEST(test_svd, fermi_symmetry) {
    auto a = (TAT::Tensor<double, TAT::FermiSymmetry>{
        {"A", "B", "C", "D"},
        {t_edge({-1, 1}, {0, 1}, {-2, 1}), f_edge({0, 1}, {1, 2}), f_edge({0, 2}, {1, 2}), t_edge({-2, 2}, {-1, 1}, {0, 2})}}
                  .range_());
    auto [u, s, v] = a.svd({"C", "A"}, "E", "F", "U", "V");
    ASSERT_NEAR(check_unitary(u, "E", "E'"), 0, 1e-8);
    ASSERT_NEAR(check_unitary(v, "F", "F'"), 0, 1e-8);
    auto b = v.contract(s, {{"F", "V"}}).contract(u, {{"U", "E"}});
    ASSERT_NEAR((a - b).norm<-1>(), 0, 1e-8);
}

TEST(test_svd, fermi_symmetry_cut) {
    auto a = (TAT::Tensor<double, TAT::FermiSymmetry>{
        {"A", "B", "C", "D"},
        {t_edge({-1, 1}, {0, 1}, {-2, 1}), f_edge({0, 1}, {1, 2}), f_edge({0, 2}, {1, 2}), t_edge({-2, 2}, {-1, 1}, {0, 2})}}
                  .range_());
    auto [u, s, v] = a.svd({"B", "D"}, "E", "F", "U", "V", 8ul);
    ASSERT_NEAR(check_unitary(u, "E", "E'"), 0, 1e-8);
    ASSERT_NEAR(check_unitary(v, "F", "F'"), 0, 1e-8);
    auto b = v.contract(s, {{"F", "V"}}).contract(u, {{"U", "E"}});
    ASSERT_NEAR((a - b).norm<-1>(), 0, 1e-8);
}

TEST(test_svd, no_symmetry_cut_too_small) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B"}, {2, 2}}.zero_();
    a.at(std::vector<TAT::Size>{0, 0}) = 1;
    auto [u, s, v] = a.svd({"B"}, "E", "F", "U", "V", 8ul);
    ASSERT_NEAR(check_unitary(u, "E", "E'"), 0, 1e-8);
    ASSERT_NEAR(check_unitary(v, "F", "F'"), 0, 1e-8);
    auto b = v.contract(s, {{"F", "V"}}).contract(u, {{"U", "E"}});
    ASSERT_NEAR((a - b).norm<-1>(), 0, 1e-8);
    ASSERT_EQ(s.storage().size(), 1);
}

TEST(test_svd, fermi_symmetry_cut_too_small) {
    auto a = TAT::Tensor<double, TAT::FermiSymmetry>{{"A", "B"}, {{{0, 1}, {+1, 1}}, {{-1, 1}, {0, 1}}}}.range_(0, 1);
    auto [u, s, v] = a.svd({"B"}, "E", "F", "U", "V", 8ul);
    ASSERT_NEAR(check_unitary(u, "E", "E'"), 0, 1e-8);
    ASSERT_NEAR(check_unitary(v, "F", "F'"), 0, 1e-8);
    auto b = v.contract(s, {{"F", "V"}}).contract(u, {{"U", "E"}});
    ASSERT_NEAR((a - b).norm<-1>(), 0, 1e-8);
    ASSERT_EQ(s.storage().size(), 1);
    ASSERT_EQ(s.edges(0).segments().size(), 1);
}
