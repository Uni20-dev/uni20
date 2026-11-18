#include "../numerics/precision_cases.hpp"
#include <uni20/linalg/dispatch_diagnostics.hpp>
#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/symmetry/block_tensor_permute.hpp>

using namespace uni20;
using namespace uni20::test;

namespace
{
template <class S, class Storage = PackedSparseBlockStorage<>>
using Matrix = BlockTensor<S, Domain<BlockSpace>, Codomain<BlockSpace>, Storage>;

struct Spaces
{
    Symmetry sym{"N:U(1)"};
    QNum q0 = QNum::identity(sym);
    BlockSpace one{sym, {{q0, 1}}, "one"};
    BlockSpace two{sym, {{q0, 1}, {make_qnum(sym, {{"N", 1}}), 1}}, "two"};
    BlockKey<2> k00{{0, 0}}, k01{{0, 1}}, k10{{1, 0}}, k11{{1, 1}};
};

TEST(BlockTensorPrecisionCoverage, ConfiguredCases)
{
  testing::Test::RecordProperty("runtime_metadata_and_exact",
                                UNI20_HAS_MPREAL ? "supported" : "unavailable: MPFR disabled");
  testing::Test::RecordProperty("finite_real_complex_128_256", UNI20_HAS_MPLAPACK_MPREAL
                                                                   ? "supported: MPLAPACK GEMM"
                                                                   : "unavailable: MPLAPACK MPFR disabled");
  testing::Test::RecordProperty("finite_diagonal_contraction", "unsupported: no finite runtime diagonal backend");
}

TEST(BlockTensorPrecision, FixedPromotionUsesResultType)
{
  Spaces g;
  Matrix<float> a(g.sym, Domain{g.one}, Codomain{g.one}, {g.k00});
  Matrix<double> b(g.sym, Domain{g.one}, Codomain{g.one}, {g.k00});
  a.block(g.k00)[0, 0] = 2;
  b.block(g.k00)[0, 0] = 3;
  auto out = contract<1, 0>(a, b);
  static_assert(std::same_as<typename decltype(out)::value_type, double>);
  EXPECT_EQ((out.block(g.k00)[0, 0]), 6);
  contract_adjacent<1>(out, a, b);
  EXPECT_EQ((out.block(g.k00)[0, 0]), 6);
  EXPECT_THROW((contract<1, 0>(out, a, b, Precision::bits(128))), std::invalid_argument);
  EXPECT_EQ((out.block(g.k00)[0, 0]), 6);
  EXPECT_THROW((Matrix<double>(g.sym, Domain{g.one}, Codomain{g.one}, {g.k00}, Precision::bits(128))),
               std::invalid_argument);
}

#if UNI20_HAS_MPREAL
template <class T>
concept WritablePrecisionDefault = requires(T& tensor, Precision p) { tensor.default_precision(p); };

TEST(BlockTensorPrecision, OwnerDefaultsSurviveViewsCopiesAndAllocation)
{
  Spaces g;
  auto const p = Precision::bits(128), q = Precision::bits(256);
  auto check = [&]<class Storage>() {
    using M = Matrix<mpreal, Storage>;
    M a(g.sym, Domain{g.one}, Codomain{g.one}, {g.k00}, p);
    EXPECT_EQ(precision_of(a), p);
    EXPECT_EQ((mpreal(a.block(g.k00)[0, 0]).precision()), p);
    EXPECT_EQ((a.block(g.k00)[0, 0]), 0);
    a.block(g.k00)[0, 0] = mpreal{7, p};
    auto block = a.block(g.k00);
    auto mapped = permute<0, 1>(permute<0, 1>(a));
    auto read_only = std::as_const(mapped).block(g.k00);
    static_assert(!WritablePrecisionDefault<decltype(block)>);
    static_assert(!WritablePrecisionDefault<decltype(mapped)>);
    a.default_precision(q);
    EXPECT_EQ(precision_of(block), q);
    EXPECT_EQ(precision_of(mapped), q);
    EXPECT_EQ(precision_of(read_only), q);
    EXPECT_EQ((mpreal(block[0, 0]).precision()), p);
    M copy = a;
    a.default_precision(p);
    EXPECT_EQ(precision_of(copy), q);
    EXPECT_EQ(precision_of(mapped), p);
    M moved = std::move(copy);
    EXPECT_EQ(precision_of(moved), q);
    M empty(g.sym, Domain{g.one}, Codomain{g.one}, std::vector<BlockKey<2>>{}, p);
    EXPECT_EQ(precision_of(empty), p);
    M unset(uninitialized, g.sym, Domain{g.one}, Codomain{g.one}, {g.k00}, p);
    EXPECT_FALSE((mpreal(unset.block(g.k00)[0, 0]).initialized()));
    EXPECT_EQ(precision_of(unset), p);
  };
  check.template operator()<PackedSparseBlockStorage<>>();
  check.template operator()<SeparateSparseBlockStorage<>>();
  check.template operator()<PackedDiagonalBlockStorage<>>();
  using Flat = BlockTensor<mpreal, Domain<BlockSpace, BlockSpace>, Codomain<>, PackedSparseBlockStorage<>>;
  Flat flat(g.sym, Domain{g.one, g.one}, Codomain<>{}, {g.k00}, p);
  auto transposed = permute<1, 0>(flat);
  auto nested = permute<1, 0>(permute<1, 0>(flat));
  flat.default_precision(q);
  EXPECT_EQ(precision_of(transposed.block(g.k00)), q);
  EXPECT_EQ(precision_of(std::as_const(nested).block(g.k00)), q);
  Matrix<mpreal, PackedCompleteBlockStorage<>> complete(g.sym, Domain{g.two}, Codomain{g.two}, p);
  EXPECT_EQ(complete.stored_block_count(), 2);
  EXPECT_EQ((complete.block(g.k11)[0, 0].precision()), p);
  Matrix<mpreal> a(g.sym, Domain{g.one}, Codomain{g.one}, {g.k00}, p);
  a.default_precision(q);
  auto zeros = a.allocate_like();
  auto unset = a.allocate_like(uninitialized);
  EXPECT_EQ(precision_of(zeros), q);
  EXPECT_EQ((zeros.block(g.k00)[0, 0].precision()), q);
  EXPECT_EQ((zeros.block(g.k00)[0, 0]), 0);
  EXPECT_EQ(precision_of(unset), q);
  EXPECT_FALSE((mpreal(unset.block(g.k00)[0, 0]).initialized()));
}

TEST(BlockTensorPrecision, ExactContractionAndExplicitPrecisionForMissingDefaults)
{
  Spaces g;
  auto check = [&]<class S>() {
    Matrix<S> a(uninitialized, g.sym, Domain{g.one}, Codomain{g.one}, {g.k00});
    Matrix<S> b(uninitialized, g.sym, Domain{g.one}, Codomain{g.one}, {g.k00});
    a.block(g.k00)[0, 0] = S{exact_rational("1/3")};
    b.block(g.k00)[0, 0] = S{exact_rational("3/7")};
    EXPECT_THROW((contract<1, 0>(a, b)), std::logic_error);
    auto out = contract<1, 0>(a, b, Precision::exact());
    EXPECT_EQ((out.block(g.k00)[0, 0]), S{exact_rational("1/7")});
    EXPECT_TRUE((out.block(g.k00)[0, 0].is_exact()));
    EXPECT_EQ(precision_of(out), Precision::exact());
    a.default_precision(Precision::exact());
    b.default_precision(Precision::exact());
    contract_adjacent<1>(out, a, b);
    EXPECT_EQ((out.block(g.k00)[0, 0]), S{exact_rational("1/7")});
  };
  check.template operator()<mpreal>();
  check.template operator()<complex<mpreal>>();
}
#endif

#if UNI20_HAS_MPLAPACK_MPREAL
using Cases = testing::Types<PrecisionCase<PrecisionKind::mp128, false>, PrecisionCase<PrecisionKind::mp128, true>,
                             PrecisionCase<PrecisionKind::mp256, false>, PrecisionCase<PrecisionKind::mp256, true>>;
template <class C> class MpfrBlockContraction : public PrecisionTest<C> {};
TYPED_TEST_SUITE(MpfrBlockContraction, Cases);

struct BlasCount
{
    int calls = 0;
    linalg::dispatch_diagnostics::scoped_sink sink{[this](auto const& event) {
      if (event.operation == "gemm")
      {
        ASSERT_TRUE(event.succeeded());
        ASSERT_TRUE(event.selected_backend());
        EXPECT_EQ(*event.selected_backend(), "blas");
        ++calls;
      }
    }};
};

TYPED_TEST(MpfrBlockContraction, SparseWorklistAccumulatesAtOnePrecision)
{
  using C = TypeParam;
  using S = typename C::scalar_type;
  Spaces g;
  auto const p = C::precision();
  auto const q = Precision::bits(32);
  S const first = C::scalar(C::real(1) + C::gap(), C::gap());
  S const expected = C::scalar(C::real(3) + C::gap(), C::gap());
  auto check = [&]<class Storage>() {
    using M = Matrix<S, Storage>;
    // Repeated local basis states are distinct legal worklist contributions;
    // BlockSpace sectors themselves have unique charges.
    LocalSpace repeated(g.sym, {g.q0, g.q0}, "repeated");
    using Left = BlockTensor<S, Domain<BlockSpace>, Codomain<LocalSpace, BlockSpace>, Storage>;
    using Right = BlockTensor<S, Domain<LocalSpace, BlockSpace>, Codomain<BlockSpace>, Storage>;
    BlockKey<3> k000{{0, 0, 0}}, k010{{0, 1, 0}}, k100{{1, 0, 0}};
    Left a(g.sym, Domain{g.two}, Codomain{repeated, g.one}, {k000, k010}, p);
    Right b(g.sym, Domain{repeated, g.one}, Codomain{g.two}, {k000, k100}, p);
    a.block(k000)[0, 0] = first;
    a.block(k010)[0, 0] = C::scalar(2);
    b.block(k000)[0, 0] = C::scalar(1);
    b.block(k100)[0, 0] = C::scalar(1);
    BlasCount count;
    auto result = contract_adjacent<2>(a, b);
    EXPECT_EQ(result.stored_block_count(), 1);
    EXPECT_EQ((result.block(g.k00)[0, 0]), expected);
    C::expect_precision(S(result.block(g.k00)[0, 0]));
    EXPECT_EQ(precision_of(result), p);
    EXPECT_EQ(count.calls, 2);
    M fixed(uninitialized, g.sym, Domain{g.two}, Codomain{g.two}, {g.k00, g.k11}, q);
    contract_adjacent<2>(fixed, a, b);
    EXPECT_EQ((fixed.block(g.k00)[0, 0]), expected);
    EXPECT_EQ((fixed.block(g.k11)[0, 0]), S{0});
    EXPECT_EQ((fixed.block(g.k11)[0, 0].precision()), p);
    EXPECT_EQ(precision_of(fixed), p);
    auto complete = contract_adjacent<2, PackedCompleteBlockStorage<>>(a, b);
    EXPECT_EQ(complete.stored_block_count(), 2);
    EXPECT_EQ((complete.block(g.k00)[0, 0]), expected);
    EXPECT_EQ((complete.block(g.k11)[0, 0]), S{0});
    EXPECT_EQ((complete.block(g.k11)[0, 0].precision()), p);
    EXPECT_EQ(count.calls, 6);

    b.default_precision(q);
    fixed.block(g.k11)[0, 0] = S{7};
    EXPECT_THROW((contract_adjacent<2>(fixed, a, b)), std::invalid_argument);
    EXPECT_EQ((fixed.block(g.k11)[0, 0]), S{7});
    // A mapped output preserves its owner's default while its values round at q.
    auto mapped = permute<0, 1>(fixed);
    contract_adjacent<2>(mapped, a, b, q);
    EXPECT_EQ((fixed.block(g.k00)[0, 0]), expected.at(q));
    EXPECT_EQ((fixed.block(g.k00)[0, 0].precision()), q);
    EXPECT_EQ(precision_of(fixed), p);
    EXPECT_EQ(count.calls, 8);
  };
  check.template operator()<PackedSparseBlockStorage<>>();
  check.template operator()<SeparateSparseBlockStorage<>>();
}

template <class Tensor> async::AsyncTask publish(async::WriteBuffer<Tensor> output, Tensor value)
{
  co_await output = std::move(value);
}

TYPED_TEST(MpfrBlockContraction, AsyncBlocksCaptureLogicalPrecisionAndAwaitValues)
{
  using C = TypeParam;
  using S = typename C::scalar_type;
  using M = Matrix<S, AsyncSeparateSparseBlockStorage<>>;
  Spaces g;
  auto const p = C::precision(), q = Precision::bits(32);
  M a(uninitialized, g.sym, Domain{g.two}, Codomain{g.one}, {g.k00}, p);
  M b(g.sym, Domain{g.one}, Codomain{g.two}, {g.k00}, p);
  b.async_block(g.k00).unsafe_value_ref()[0, 0] = C::scalar(1);
  auto first = C::matrix(1, 1);
  first[0, 0] = C::scalar(C::real(1) + C::gap(), C::gap());
  S const expected = C::scalar(C::real(1) + C::gap(), C::gap());
  async::DebugScheduler scheduler;
  async::ScopedScheduler scoped(&scheduler);
  auto first_write = a.async_block(g.k00).write();
  BlasCount count;
  auto out = contract<1, 0>(a, b);
  a.default_precision(q); // Already submitted work retains its selected precision.
  scheduler.schedule(publish(std::move(first_write), std::move(first)));
  auto const& value = out.async_block(g.k00).get_wait(scheduler);
  EXPECT_EQ((value[0, 0]), expected);
  EXPECT_EQ(value.default_precision(), p);
  EXPECT_EQ(precision_of(out), p);
  EXPECT_EQ(count.calls, 1);
  contract<1, 0>(out, a, b, q);
  auto const& rounded = out.async_block(g.k00).get_wait(scheduler);
  EXPECT_EQ((rounded[0, 0]), expected.at(q));
  EXPECT_EQ(precision_of(out), q);
  EXPECT_EQ((rounded[0, 0].precision()), q);
  EXPECT_EQ(count.calls, 2);
  M fixed(uninitialized, g.sym, Domain{g.two}, Codomain{g.two}, {g.k00, g.k11}, q);
  contract<1, 0>(fixed, a, b, p);
  auto const& zero = fixed.async_block(g.k11).get_wait(scheduler);
  EXPECT_EQ((zero[0, 0]), S{0});
  EXPECT_EQ((zero[0, 0].precision()), p);
  EXPECT_EQ(zero.default_precision(), p);
  EXPECT_EQ((fixed.async_block(g.k00).get_wait(scheduler)[0, 0]), expected);
  EXPECT_EQ(count.calls, 3);
}
#endif
} // namespace
