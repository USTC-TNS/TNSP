#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

#ifdef TAT_USE_MPI

TEST(test_mpi, basic_usage) {
    auto rank = TAT::mpi.rank;
    auto size = TAT::mpi.size;
    TAT::mpi.barrier();
}

TEST(test_mpi, send_and_recv) {
    auto rank = TAT::mpi.rank;
    auto size = TAT::mpi.size;
    if (size > 1) {
        auto a = TAT::Tensor<>({"i", "j"}, {2, 3}).zero_();
        if (rank == 0) {
            a.range_();
            TAT::mpi.send(a, 1);
        }
        if (rank == 1) {
            auto b = TAT::mpi.receive<TAT::Tensor<>>(0);
            ASSERT_FLOAT_EQ((a.range_() - b).norm<-1>(), 0);
        }
    }
}

TEST(test_mpi, send_receive) {
    auto rank = TAT::mpi.rank;
    auto size = TAT::mpi.size;
    if (size > 1) {
        auto a = TAT::Tensor<>({"i", "j"}, {2, 3}).range_();
        auto b = TAT::mpi.send_receive(a, 0, 1);
        if (rank == 1) {
            ASSERT_FLOAT_EQ((a - b).norm<-1>(), 0);
        } else {
            ASSERT_EQ(b.storage().size(), 1);
        }
    }
}

TEST(test_mpi, broadcast) {
    auto rank = TAT::mpi.rank;
    auto size = TAT::mpi.size;
    auto a = TAT::Tensor<>({"i", "j"}, {2, 3}).range_();
    auto b = TAT::Tensor<>({"i", "j"}, {2, 3}).range_(1);
    auto c = TAT::Tensor<>({"i", "j"}, {2, 3}).zero_();
    if (rank == 0) {
        c = a + b;
    }
    c = TAT::mpi.broadcast(c, 0);
    ASSERT_FLOAT_EQ((a + b - c).norm<-1>(), 0);
}

TEST(test_mpi, reduce) {
    auto rank = TAT::mpi.rank;
    auto size = TAT::mpi.size;
    auto a = TAT::Tensor<>({}, {}).range_(rank);
    auto b = TAT::mpi.reduce(a, 0, [](const auto& a, const auto& b) { return a + b; });
    if (rank == 0) {
        ASSERT_FLOAT_EQ(double(b), (size - 1) * size / 2.);
    }
}

#endif
