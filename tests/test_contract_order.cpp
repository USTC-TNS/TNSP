#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

// The current strategy for matrix matrix product is:
// If tensor_1 is larger than tensor_2, check tensor_1 first, otherwise check tensor_2 first.
// If free_name_1[-1] != name_1[-1], aka, tensor_1 last edge will be contracted:
//   Fit common name by tensor1
//   Put common of tensor1 to right
//   Whether to put common of tensor2 to right -> common_name_2[-1] == name_2[-1]
// If free_name_2[-1] != name_1[-1], aka, tensor_2 last edge will be contracted:
//   Fit common name by tensor2
//   Put common of tensor2 to right
//   Whether to put common of tensor1 to right -> common_name_1[-1] == name_1[-1]
// Otherwise, last edge of both tensor is free edge.
//   Put common name to left for both
//   Fit which tensor? the large one's common edge will be considered as fit plan.
// For matrix vector product, The strategy is:
//   Fit the common edge by matrix's common edge since it is larger.
//   Put common of matrix to right? -> common_name_matrix[-1] == name_matrix[-1]
TEST(test_contract_order, no_symmetry_matrix_matrix_better_left_by_left) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B"}, {3, 2}}.range();
    auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"C", "D"}, {2, 2}}.range();
    auto c = TAT::Tensor<double, TAT::NoSymmetry>::contract(a, b, {{"B", "C"}});
}

TEST(test_contract_order, no_symmetry_matrix_matrix_better_left_by_right) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B"}, {2, 2}}.range();
    auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"C", "D"}, {2, 3}}.range();
    auto c = TAT::Tensor<double, TAT::NoSymmetry>::contract(a, b, {{"B", "C"}});
}

TEST(test_contract_order, no_symmetry_matrix_matrix_better_right_by_left) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B"}, {2, 3}}.range();
    auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"C", "D"}, {2, 2}}.range();
    auto c = TAT::Tensor<double, TAT::NoSymmetry>::contract(a, b, {{"A", "D"}});
}

TEST(test_contract_order, no_symmetry_matrix_matrix_better_right_by_right) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B"}, {2, 2}}.range();
    auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"C", "D"}, {3, 2}}.range();
    auto c = TAT::Tensor<double, TAT::NoSymmetry>::contract(a, b, {{"A", "D"}});
}

TEST(test_contract_order, no_symmetry_matrix_matrix_both_bad_by_left) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B"}, {2, 3}}.range();
    auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"C", "D"}, {2, 2}}.range();
    auto c = TAT::Tensor<double, TAT::NoSymmetry>::contract(a, b, {{"A", "C"}});
}

TEST(test_contract_order, no_symmetry_matrix_matrix_both_bad_by_right) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B"}, {2, 2}}.range();
    auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"C", "D"}, {2, 3}}.range();
    auto c = TAT::Tensor<double, TAT::NoSymmetry>::contract(a, b, {{"A", "C"}});
}

TEST(test_contract_order, no_symmetry_matrix_vector) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A", "B"}, {2, 2}}.range();
    auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"C"}, {2}}.range();
    auto c = TAT::Tensor<double, TAT::NoSymmetry>::contract(a, b, {{"A", "C"}});
}

TEST(test_contract_order, no_symmetry_vector_matrix) {
    auto a = TAT::Tensor<double, TAT::NoSymmetry>{{"A"}, {2}}.range();
    auto b = TAT::Tensor<double, TAT::NoSymmetry>{{"C", "D"}, {2, 2}}.range();
    auto c = TAT::Tensor<double, TAT::NoSymmetry>::contract(a, b, {{"A", "C"}});
}

TEST(test_contract_order, z2_symmetry_matrix_matrix_better_left_by_left) {
    auto a = TAT::Tensor<double, TAT::Z2Symmetry>{{"A", "B"}, {{{false, 3}, {true, 3}}, {{false, 2}, {true, 2}}}}.range();
    auto b = TAT::Tensor<double, TAT::Z2Symmetry>{{"C", "D"}, {{{false, 2}, {true, 2}}, {{false, 2}, {true, 2}}}}.range();
    auto c = TAT::Tensor<double, TAT::Z2Symmetry>::contract(a, b, {{"B", "C"}});
}

TEST(test_contract_order, z2_symmetry_matrix_matrix_better_left_by_right) {
    auto a = TAT::Tensor<double, TAT::Z2Symmetry>{{"A", "B"}, {{{false, 2}, {true, 2}}, {{false, 2}, {true, 2}}}}.range();
    auto b = TAT::Tensor<double, TAT::Z2Symmetry>{{"C", "D"}, {{{false, 2}, {true, 2}}, {{false, 3}, {true, 3}}}}.range();
    auto c = TAT::Tensor<double, TAT::Z2Symmetry>::contract(a, b, {{"B", "C"}});
}

TEST(test_contract_order, z2_symmetry_matrix_matrix_better_right_by_left) {
    auto a = TAT::Tensor<double, TAT::Z2Symmetry>{{"A", "B"}, {{{false, 2}, {true, 2}}, {{false, 3}, {true, 3}}}}.range();
    auto b = TAT::Tensor<double, TAT::Z2Symmetry>{{"C", "D"}, {{{false, 2}, {true, 2}}, {{false, 2}, {true, 2}}}}.range();
    auto c = TAT::Tensor<double, TAT::Z2Symmetry>::contract(a, b, {{"A", "D"}});
}

TEST(test_contract_order, z2_symmetry_matrix_matrix_better_right_by_right) {
    auto a = TAT::Tensor<double, TAT::Z2Symmetry>{{"A", "B"}, {{{false, 2}, {true, 2}}, {{false, 2}, {true, 2}}}}.range();
    auto b = TAT::Tensor<double, TAT::Z2Symmetry>{{"C", "D"}, {{{false, 3}, {true, 3}}, {{false, 2}, {true, 2}}}}.range();
    auto c = TAT::Tensor<double, TAT::Z2Symmetry>::contract(a, b, {{"A", "D"}});
}

TEST(test_contract_order, z2_symmetry_matrix_matrix_both_bad_by_left) {
    auto a = TAT::Tensor<double, TAT::Z2Symmetry>{{"A", "B"}, {{{false, 2}, {true, 2}}, {{false, 3}, {true, 3}}}}.range();
    auto b = TAT::Tensor<double, TAT::Z2Symmetry>{{"C", "D"}, {{{false, 2}, {true, 2}}, {{false, 2}, {true, 2}}}}.range();
    auto c = TAT::Tensor<double, TAT::Z2Symmetry>::contract(a, b, {{"A", "C"}});
}

TEST(test_contract_order, z2_symmetry_matrix_matrix_both_bad_by_right) {
    auto a = TAT::Tensor<double, TAT::Z2Symmetry>{{"A", "B"}, {{{false, 2}, {true, 2}}, {{false, 2}, {true, 2}}}}.range();
    auto b = TAT::Tensor<double, TAT::Z2Symmetry>{{"C", "D"}, {{{false, 2}, {true, 2}}, {{false, 3}, {true, 3}}}}.range();
    auto c = TAT::Tensor<double, TAT::Z2Symmetry>::contract(a, b, {{"A", "C"}});
}

TEST(test_contract_order, z2_symmetry_matrix_vector) {
    auto a = TAT::Tensor<double, TAT::Z2Symmetry>{{"A", "B"}, {{{false, 2}, {true, 2}}, {{false, 2}, {true, 2}}}}.range();
    auto b = TAT::Tensor<double, TAT::Z2Symmetry>{{"C"}, {{{false, 2}, {true, 2}}}}.range();
    auto c = TAT::Tensor<double, TAT::Z2Symmetry>::contract(a, b, {{"A", "C"}});
}

TEST(test_contract_order, z2_symmetry_vector_matrix) {
    auto a = TAT::Tensor<double, TAT::Z2Symmetry>{{"A"}, {{{false, 2}, {true, 2}}}}.range();
    auto b = TAT::Tensor<double, TAT::Z2Symmetry>{{"C", "D"}, {{{false, 2}, {true, 2}}, {{false, 2}, {true, 2}}}}.range();
    auto c = TAT::Tensor<double, TAT::Z2Symmetry>::contract(a, b, {{"A", "C"}});
}
