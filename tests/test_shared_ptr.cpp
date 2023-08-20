#include <TAT/TAT.hpp>
#include <gtest/gtest.h>

TEST(test_shared_ptr, basic_usage) {
   auto p1 = TAT::detail::shared_ptr<int>::make(10);
   ASSERT_EQ(p1.use_count(), 1);
   auto p2 = p1;
   ASSERT_EQ(p1.use_count(), 2);
   ASSERT_EQ(p2.use_count(), 2);
   auto p3 = std::move(p1);
   ASSERT_EQ(p1.use_count(), 0);
   ASSERT_EQ(p2.use_count(), 2);
   ASSERT_EQ(p3.use_count(), 2);
   p1 = p2;
   ASSERT_EQ(p1.use_count(), 3);
   ASSERT_EQ(p2.use_count(), 3);
   ASSERT_EQ(p3.use_count(), 3);
   p2 = std::move(p3);
   ASSERT_EQ(p1.use_count(), 2);
   ASSERT_EQ(p2.use_count(), 2);
   ASSERT_EQ(p3.use_count(), 0);
}

TEST(test_shared_ptr, get_object) {
   struct V {
      int v;
      explicit V(int v) : v(v) {}
   };

   auto p = TAT::detail::shared_ptr<V>::make(10);
   ASSERT_EQ(p->v, 10);
   ASSERT_EQ(p.get()->v, 10);
   ASSERT_EQ((*p).v, 10);

   const auto q = TAT::detail::shared_ptr<V>::make(20);
   ASSERT_EQ(q->v, 20);
   ASSERT_EQ(q.get()->v, 20);
   ASSERT_EQ((*q).v, 20);
}
