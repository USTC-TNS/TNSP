#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

auto factor(int i) {
    if (i == 0) {
        return 1;
    } else {
        return i * factor(i - 1);
    }
}

template<typename T, typename N>
auto reference_exponential(const T& tensor, const N& pairs, int step) {
    auto result = tensor.same_shape().identity(pairs);
    auto power = result;
    for (auto i = 1; i < step; i++) {
        power = power.contract(tensor, pairs) / i;
        result += power;
    }
    return result;
}

TEST(test_exponential, no_symmetry) {
    auto A = TAT::Tensor<double, TAT::NoSymmetry>({"i", "j"}, {3, 3}).range();
    auto pairs = std::unordered_set<std::pair<std::string, std::string>>{{"i", "j"}};
    auto expA = A.exponential(pairs, 10);
    auto expA_r = reference_exponential(A, pairs, 100);
    ASSERT_NEAR((expA - expA_r).norm<-1>(), 0, 1e-8);
}

TEST(test_exponential, u1_symmetry) {
    auto A = (TAT::Tensor<double, TAT::U1Symmetry>(
                  {"i", "j", "k", "l"},
                  {{{-1, 2}, {0, 2}, {+1, 2}}, {{+1, 2}, {0, 2}, {-1, 2}}, {{+1, 2}, {0, 2}, {-1, 2}}, {{-1, 2}, {0, 2}, {+1, 2}}}
    ).range());
    A /= A.norm<0>();
    auto pairs = std::unordered_set<std::pair<std::string, std::string>>{{"i", "k"}, {"l", "j"}};
    auto expA = A.exponential(pairs, 10);
    auto expA_r = reference_exponential(A, pairs, 100);
    ASSERT_NEAR((expA - expA_r).norm<-1>(), 0, 1e-8);
}

TEST(test_exponential, fermi_symmetry) {
    auto A = (TAT::Tensor<double, TAT::FermiSymmetry>(
                  {"i", "j", "k", "l"},
                  {
                      {{{-1, 2}, {0, 2}, {+1, 2}}, true},
                      {{{+1, 2}, {0, 2}, {-1, 2}}, true},
                      {{{+1, 2}, {0, 2}, {-1, 2}}, false},
                      {{{-1, 2}, {0, 2}, {+1, 2}}, false},
                  }
    )
                  .range());
    A /= A.norm<0>();
    auto pairs = std::unordered_set<std::pair<std::string, std::string>>{{"i", "k"}, {"l", "j"}};
    auto expA = A.exponential(pairs, 10);
    auto expA_r = reference_exponential(A, pairs, 100);
    ASSERT_NEAR((expA - expA_r).norm<-1>(), 0, 1e-8);
}
