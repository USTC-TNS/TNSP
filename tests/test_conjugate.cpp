#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_conjugate, no_symmetry_float) {
   auto A = TAT::Tensor<double, TAT::NoSymmetry>({"i", "j"}, {2, 3}).range();
   auto A_c = A.conjugate();
   ASSERT_EQ(&A.storage(), &A_c.storage());
}

TEST(test_conjugate, no_symmetry_complex) {
   auto A = TAT::Tensor<std::complex<double>, TAT::NoSymmetry>({"i", "j"}, {2, 3}).range({1, 5}, {1, 7});
   auto A_c = A.conjugate();
   auto B = A * A_c;
   for (auto i : B.storage()) {
      ASSERT_FLOAT_EQ(i.imag(), 0);
   }
}

TEST(test_conjugate, u1_symmetry_float) {
   auto A = (TAT::Tensor<double, TAT::U1Symmetry>({"i", "j"}, {{{-1, 2}, {0, 2}, {+1, 2}}, {{-1, 2}, {0, 2}, {+1, 2}}}).range(-8, 1));
   auto A_c = A.conjugate();
   auto B = A.contract(A_c, {{"i", "i"}, {"j", "j"}});
   ASSERT_GT(double(B), 0);
   ASSERT_FLOAT_EQ((A_c.conjugate() - A).norm<-1>(), 0);
}

TEST(test_conjugate, u1_symmetry_complex) {
   auto A = (TAT::Tensor<std::complex<double>, TAT::U1Symmetry>({"i", "j"}, {{{-1, 2}, {0, 2}, {+1, 2}}, {{-1, 2}, {0, 2}, {+1, 2}}})
                   .range({-8, -20}, {1, 7}));
   auto A_c = A.conjugate();
   auto B = A.contract(A_c, {{"i", "i"}, {"j", "j"}});
   ASSERT_FLOAT_EQ(std::complex<double>(B).imag(), 0);
   ASSERT_GT(std::complex<double>(B).real(), 0);
   ASSERT_FLOAT_EQ((A_c.conjugate() - A).norm<-1>(), 0);
}

TEST(test_conjugate, fermi_symmmtry_float) {
   auto A = (TAT::Tensor<double, TAT::FermiSymmetry>({"i", "j"}, {{{-1, 2}, {0, 2}, {+1, 2}}, {{-1, 2}, {0, 2}, {+1, 2}}}).range(-8, 1));
   auto A_c = A.conjugate();
   auto B = A.contract(A_c, {{"i", "i"}, {"j", "j"}});
   ASSERT_GT(double(B), 0);
   ASSERT_FLOAT_EQ((A_c.conjugate() - A).norm<-1>(), 0);
}

TEST(test_conjugate, fermi_symmmtry_float_bidirection_arrow) {
   auto A = (TAT::Tensor<double, TAT::FermiSymmetry>({"i", "j"}, {{{{-1, 2}, {+1, 2}}, false}, {{{-1, 2}, {+1, 2}}, true}}).range(-8, 1));
   auto A_c = A.conjugate();
   auto B = A.contract(A_c, {{"i", "i"}, {"j", "j"}});
   ASSERT_LE(double(B), 0); // The A * Ac may not be positive
   ASSERT_FLOAT_EQ((A_c.conjugate() - A).norm<-1>(), 0);
}

TEST(test_conjugate, fermi_symmmtry_float_bidirection_arrow_fixed) {
   auto A = (TAT::Tensor<double, TAT::FermiSymmetry>({"i", "j"}, {{{{-1, 2}, {+1, 2}}, false}, {{{-1, 2}, {+1, 2}}, true}}).range(-8, 1));
   auto A_c = A.conjugate(true);
   auto B = A.contract(A_c, {{"i", "i"}, {"j", "j"}});
   ASSERT_GE(double(B), 0);
   ASSERT_FLOAT_EQ((A_c.conjugate(true) - A).norm<-1>(), 0);
}

TEST(test_conjugate, fermi_symmmtry_complex) {
   auto A = (TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>({"i", "j"}, {{{-1, 2}, {0, 2}, {+1, 2}}, {{-1, 2}, {0, 2}, {+1, 2}}})
                   .range({-8, -20}, {1, 7}));
   auto A_c = A.conjugate();
   auto B = A.contract(A_c, {{"i", "i"}, {"j", "j"}});
   ASSERT_GT(std::complex<double>(B).real(), 0);
   ASSERT_FLOAT_EQ(std::complex<double>(B).imag(), 0);
   ASSERT_FLOAT_EQ((A_c.conjugate() - A).norm<-1>(), 0);
}

TEST(test_conjugate, fermi_symmmtry_complex_bidirection_arrow) {
   auto A = (TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>({"i", "j"}, {{{{-1, 2}, {+1, 2}}, false}, {{{-1, 2}, {+1, 2}}, true}})
                   .range({-8, -20}, {1, 7}));
   auto A_c = A.conjugate();
   auto B = A.contract(A_c, {{"i", "i"}, {"j", "j"}});
   ASSERT_LE(std::complex<double>(B).real(), 0);       // The A * Ac may not be positive
   ASSERT_FLOAT_EQ(std::complex<double>(B).imag(), 0); // It is also a real number
   ASSERT_FLOAT_EQ((A_c.conjugate() - A).norm<-1>(), 0);
}

TEST(test_conjugate, fermi_symmmtry_complex_bidirection_arrow_fixed) {
   auto A = (TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>({"i", "j"}, {{{{-1, 2}, {+1, 2}}, false}, {{{-1, 2}, {+1, 2}}, true}})
                   .range({-8, -20}, {1, 7}));
   auto A_c = A.conjugate(true);
   auto B = A.contract(A_c, {{"i", "i"}, {"j", "j"}});
   ASSERT_GE(std::complex<double>(B).real(), 0);
   ASSERT_FLOAT_EQ(std::complex<double>(B).imag(), 0);
   ASSERT_FLOAT_EQ((A_c.conjugate(true) - A).norm<-1>(), 0);
}

TEST(test_conjugate, fermi_symmetry_contract_with_conjugate) {
   auto A = (TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>{
         {"i", "j"},
         {
               {{{-1, 2}, {0, 2}, {+1, 2}}, false},
               {{{+1, 2}, {0, 2}, {-1, 2}}, true},
         }}.range({-8, -20}, {1, 7}));
   auto B = (TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>{
         {"i", "j"},
         {
               {{{-1, 2}, {0, 2}, {+1, 2}}, false},
               {{{+1, 2}, {0, 2}, {-1, 2}}, true},
         }}.range({-7, -29}, {5, 3}));
   auto C = A.contract(B, {{"i", "j"}});
   auto A_c = A.conjugate();
   auto B_c = B.conjugate();
   auto C_c = A_c.contract(B_c, {{"i", "j"}});
   ASSERT_FLOAT_EQ((C.conjugate() - C_c).norm<-1>(), 0);
}

TEST(test_conjugate, fermi_symmetry_contract_with_conjugate_arrow_fix_wrong) {
   auto A = (TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>{
         {"i", "j"},
         {
               {{{-1, 2}, {0, 2}, {+1, 2}}, false},
               {{{+1, 2}, {0, 2}, {-1, 2}}, true},
         }}.range({-8, -20}, {1, 7}));
   auto B = (TAT::Tensor<std::complex<double>, TAT::FermiSymmetry>{
         {"i", "j"},
         {
               {{{-1, 2}, {0, 2}, {+1, 2}}, false},
               {{{+1, 2}, {0, 2}, {-1, 2}}, true},
         }}.range({-7, -29}, {5, 3}));
   auto C = A.contract(B, {{"i", "j"}});
   auto A_c = A.conjugate(true);
   auto B_c = B.conjugate(true);
   auto C_c = A_c.contract(B_c, {{"i", "j"}});
   ASSERT_GE((C.conjugate(true) - C_c).norm<-1>(), 1e-8);
}
