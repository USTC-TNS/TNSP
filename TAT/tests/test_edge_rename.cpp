#include <TAT/TAT.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace testing;

TEST(test_edge_rename, basic_rename) {
    auto t1 = TAT::Tensor<double, TAT::BoseZ2Symmetry>{{"Left", "Right", "Phy"}, {{{0, 1}, {1, 2}}, {{0, 3}, {1, 4}}, {{0, 5}, {1, 6}}}}.range_();
    auto t2 = t1.edge_rename({{"Left", "Up"}});
    ASSERT_THAT(t1.names(), ElementsAre("Left", "Right", "Phy"));
    ASSERT_THAT(t2.names(), ElementsAre("Up", "Right", "Phy"));
    ASSERT_EQ(&t2.storage(), &t1.storage());
}

namespace net {
    using pss = std::tuple<std::string, std::string>;

    std::ostream& operator<<(std::ostream& os, const pss& p) {
        return os << std::get<0>(p) << "." << std::get<1>(p);
    }
} // namespace net

namespace std {
    template<>
    struct hash<net::pss> {
        size_t operator()(const net::pss& name) const {
            std::hash<std::string> string_hash;
            return string_hash(std::get<0>(name)) ^ !string_hash(std::get<1>(name));
        }
    };
} // namespace std

namespace TAT {
    template<>
    const net::pss InternalName<net::pss>::Default_0 = {"Internal", "0"};
    template<>
    const net::pss InternalName<net::pss>::Default_1 = {"Internal", "1"};
    template<>
    const net::pss InternalName<net::pss>::Default_2 = {"Internal", "2"};
    template<>
    const net::pss InternalName<net::pss>::Default_3 = {"Internal", "3"};
    template<>
    const net::pss InternalName<net::pss>::Default_4 = {"Internal", "4"};

    template<>
    struct NameTraits<net::pss> {
        static constexpr out_operator_t<net::pss> print = net::operator<<;
    };
} // namespace TAT

TEST(test_edge_rename, customed_name) {
    std::stringstream si, sd;
    TAT::NameTraits<net::pss>::print(sd, TAT::InternalName<net::pss>::SVD_U);
    TAT::NameTraits<net::pss>::print(si, TAT::InternalName<net::pss>::Default_1);
    TAT::NameTraits<net::pss>::print(sd, TAT::InternalName<net::pss>::QR_2);
    TAT::NameTraits<net::pss>::print(si, TAT::InternalName<net::pss>::Default_2);
    TAT::NameTraits<net::pss>::print(sd, TAT::InternalName<net::pss>::Contract_0);
    TAT::NameTraits<net::pss>::print(si, TAT::InternalName<net::pss>::Default_0);
    TAT::NameTraits<net::pss>::print(sd, TAT::InternalName<net::pss>::No_Old_Name);
    TAT::NameTraits<net::pss>::print(si, TAT::InternalName<net::pss>::Default_0);
    TAT::NameTraits<net::pss>::print(sd, TAT::InternalName<net::pss>::Exp_2);
    TAT::NameTraits<net::pss>::print(si, TAT::InternalName<net::pss>::Default_2);
    TAT::NameTraits<net::pss>::print(sd, TAT::InternalName<net::pss>::Trace_5);
    TAT::NameTraits<net::pss>::print(si, TAT::InternalName<net::pss>::Default_4);
    ASSERT_EQ(si.str(), sd.str());
}

TEST(test_edge_rename, rename_with_customed_name) {
    auto t1 = TAT::Tensor<double, TAT::NoSymmetry, net::pss>({{"A", "1"}, {"A", "2"}}, {5, 5}).range_();
    auto t2 = t1.edge_rename(std::unordered_map<net::pss, std::string>{{{"A", "1"}, "A1"}, {{"A", "2"}, "A2"}});
    ASSERT_THAT(t2.names(), ElementsAre("A1", "A2"));
    ASSERT_EQ(&t2.storage(), &t1.storage());
}
