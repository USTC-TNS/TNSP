#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

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
    auto identity = product.same_shape().identity({{name, name_prime}});
    if constexpr (T::symmetry_t::is_fermi_symmetry) {
        // some sign maybe wrong
        product.transform([](auto x) { return std::abs(x); });
        identity.transform([](auto x) { return std::abs(x); });
    }
    return (product - identity).template norm<-1>();
}

TEST(test_qr, no_symmetry_0) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>({"A", "B"}, {5, 10}).range();
    auto [q, r] = a.qr('r', {"A"}, "newQ", "newR");
    auto [q_2, r_2] = a.qr('q', {"B"}, "newQ", "newR");
    ASSERT_FLOAT_EQ((q - q_2).norm<-1>(), 0);
    ASSERT_FLOAT_EQ((r - r_2).norm<-1>(), 0);
    ASSERT_THAT(q.names(), ElementsAre("newQ", "B"));
    ASSERT_THAT(r.names(), ElementsAre("A", "newR"));
    ASSERT_EQ(q.edges("newQ").total_dimension(), 5);
    ASSERT_EQ(r.edges("newR").total_dimension(), 5);
    ASSERT_NEAR(check_unitary(q, "newQ", "newQ'"), 0, 1e-8);
    ASSERT_NEAR((q.contract(r, {{"newQ", "newR"}}) - a).norm<-1>(), 0, 1e-8);
}

TEST(test_qr, no_symmetry_1) {
    auto a = TAT::Tensor<std::complex<double>, TAT::NoSymmetry>({"A", "B"}, {10, 5}).range({-21, -29}, {+2, +3});
    auto [q, r] = a.qr('q', {"A"}, "newQ", "newR");
    auto [q_2, r_2] = a.qr('r', {"B"}, "newQ", "newR");
    ASSERT_FLOAT_EQ((q - q_2).norm<-1>(), 0);
    ASSERT_FLOAT_EQ((r - r_2).norm<-1>(), 0);
    ASSERT_THAT(q.names(), ElementsAre("A", "newQ"));
    ASSERT_THAT(r.names(), ElementsAre("newR", "B"));
    ASSERT_EQ(q.edges("newQ").total_dimension(), 5);
    ASSERT_EQ(r.edges("newR").total_dimension(), 5);
    ASSERT_NEAR(check_unitary(q, "newQ", "newQ'"), 0, 1e-8);
    ASSERT_NEAR((q.contract(r, {{"newQ", "newR"}}) - a).norm<-1>(), 0, 1e-8);
}

TEST(test_qr, fermi_symmetry_0) {
    auto a = TAT::Tensor<double, TAT::FermiSymmetry>({"A", "B"}, {{{{-1, 2}, {0, 1}, {+1, 2}}, false}, {{{-1, 4}, {0, 3}, {+1, 3}}, true}}).range();
    auto [q, r] = a.qr('r', {"A"}, "newQ", "newR");
    auto [q_2, r_2] = a.qr('q', {"B"}, "newQ", "newR");
    ASSERT_FLOAT_EQ((q - q_2).norm<-1>(), 0);
    ASSERT_FLOAT_EQ((r - r_2).norm<-1>(), 0);
    ASSERT_THAT(q.names(), ElementsAre("newQ", "B"));
    ASSERT_THAT(r.names(), ElementsAre("A", "newR"));
    ASSERT_EQ(q.edges("newQ").total_dimension(), 5);
    ASSERT_EQ(r.edges("newR").total_dimension(), 5);
    ASSERT_NEAR(check_unitary(q, "newQ", "newQ'"), 0, 1e-8);
    ASSERT_NEAR((q.contract(r, {{"newQ", "newR"}}) - a).norm<-1>(), 0, 1e-8);
}

TEST(test_qr, fermi_symmetry_1) {
    auto a = (TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>(
                  {"A", "B"},
                  {
                      {{{-1, 2}, {0, 1}, {+1, 2}}, true},
                      {{{-1, 4}, {0, 3}, {+1, 3}}, false},
                  }
    )
                  .range());
    auto [q, r] = a.qr('q', {"A"}, "newQ", "newR");
    auto [q_2, r_2] = a.qr('r', {"B"}, "newQ", "newR");
    ASSERT_FLOAT_EQ((q - q_2).norm<-1>(), 0);
    ASSERT_FLOAT_EQ((r - r_2).norm<-1>(), 0);
    ASSERT_THAT(q.names(), ElementsAre("A", "newQ"));
    ASSERT_THAT(r.names(), ElementsAre("newR", "B"));
    ASSERT_EQ(q.edges("newQ").total_dimension(), 5);
    ASSERT_EQ(r.edges("newR").total_dimension(), 5);
    ASSERT_NEAR(check_unitary(q, "newQ", "newQ'"), 0, 1e-8);
    ASSERT_NEAR((q.contract(r, {{"newQ", "newR"}}) - a).norm<-1>(), 0, 1e-8);
}

TEST(test_qr, fermi_symmetry_edge_mismatch) {
    auto a = (TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>(
                  {"A", "B"},
                  {
                      {{{-1, 2}, {0, 2}, {+2, 2}}, false},
                      {{{-1, 4}, {0, 3}, {+2, 3}}, true},
                  }
    )
                  .range());
    auto [q, r] = a.qr('q', {"A"}, "newQ", "newR");
    auto [q_2, r_2] = a.qr('r', {"B"}, "newQ", "newR");
    ASSERT_FLOAT_EQ((q - q_2).norm<-1>(), 0);
    ASSERT_FLOAT_EQ((r - r_2).norm<-1>(), 0);
    ASSERT_THAT(q.names(), ElementsAre("A", "newQ"));
    ASSERT_THAT(r.names(), ElementsAre("newR", "B"));
    ASSERT_EQ(q.edges("newQ").total_dimension(), 2);
    ASSERT_EQ(r.edges("newR").total_dimension(), 2);
    ASSERT_NEAR(check_unitary(q, "newQ", "newQ'"), 0, 1e-8);
    ASSERT_NEAR((q.contract(r, {{"newQ", "newR"}}) - a).norm<-1>(), 0, 1e-8);
}
