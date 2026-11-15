#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/symmetry/block_tensor_linear.hpp>
#include <uni20/symmetry/block_tensor_permute.hpp>
#include <uni20/symmetry/block_tensor_repartition.hpp>
#include <uni20/symmetry/block_tensor_svd.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using namespace uni20;

template <class Block, class Function> void for_each_block_index(Block const& block, Function&& function)
{
  constexpr std::size_t rank = Block::rank();
  std::array<index_type, rank> indices{};
  auto visit = [&](auto&& self, std::size_t axis) -> void {
    if (axis == rank)
    {
      std::apply(function, indices);
      return;
    }
    for (index_type i = 0; i < block.extent(axis); ++i)
    {
      indices[axis] = i;
      self(self, axis + 1);
    }
  };
  visit(visit, 0);
}

template <class Storage> class HighOrderBlockTensorTest : public ::testing::Test {};
using StorageTypes = ::testing::Types<SeparateSparseBlockStorage<>, PackedSparseBlockStorage<>>;
TYPED_TEST_SUITE(HighOrderBlockTensorTest, StorageTypes);

TYPED_TEST(HighOrderBlockTensorTest, ContractsFourLegTensorsAndPreservesSixLegViewMetadata)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  BlockSpace const a(sym, {{q0, 1}, {q1, 2}}, "a");
  BlockSpace const b(sym, {{q0, 2}}, "b");
  BlockSpace const c(sym, {{q0, 3}}, "c");
  BlockSpace const bond(sym, {{q0, 2}, {q1, 3}}, "bond");
  BlockSpace const d(sym, {{q0, 2}}, "d");
  BlockSpace const e(sym, {{q0, 2}, {q1, 1}}, "e");
  BlockSpace const f(sym, {{q0, 2}}, "f");
  using Tensor = BlockTensor<double, Domain<BlockSpace, BlockSpace>, Codomain<BlockSpace, BlockSpace>, TypeParam>;
  using Key = typename Tensor::key_type;
  Tensor left(sym, Domain{a, b}, Codomain{c, bond}, {Key{{0, 0, 0, 0}}, Key{{1, 0, 0, 1}}});
  Tensor right(sym, Domain{bond, d}, Codomain{e, f}, {Key{{0, 0, 0, 0}}, Key{{1, 0, 1, 0}}});
  for (std::size_t sector = 0; sector < 2; ++sector)
  {
    auto lhs = left.block(Key{{sector, 0, 0, sector}});
    auto rhs = right.block(Key{{sector, 0, sector, 0}});
    for_each_block_index(lhs, [&](auto i, auto j, auto k, auto s) {
      lhs[i, j, k, s] = static_cast<double>(1 + i + 2 * j + 3 * k + 5 * s + 7 * sector);
    });
    for_each_block_index(rhs, [&](auto s, auto l, auto m, auto n) {
      rhs[s, l, m, n] = static_cast<double>(2 + 3 * s + 5 * l + 7 * m + n + 11 * sector);
    });
  }

  auto result = contract<3, 0>(left, right);
  using Result = decltype(result);
  using ResultKey = typename Result::key_type;
  static_assert(Result::order() == 6);
  static_assert(Result::dense_block_order() == 6);
  ASSERT_EQ(result.stored_block_count(), 2);
  EXPECT_EQ(result.domain(), (Domain{a, b, d}));
  EXPECT_EQ(result.codomain(), (Codomain{c, e, f}));
  EXPECT_FALSE(result.is_legal(ResultKey{{0, 0, 0, 0, 1, 0}}));

  // Reorder nontrivial, unequal extents on both sides and materialize the view.
  auto view = permute<2, 1, 0, 4, 5, 3>(result);
  using Copy =
      BlockTensor<double, typename decltype(view)::domain_type, typename decltype(view)::codomain_type, TypeParam>;
  Copy copied(sym, view.domain(), view.codomain(),
              std::vector<ResultKey>(view.stored_keys().begin(), view.stored_keys().end()));
  copy(copied, view);
  auto bent = repartition<MorphismSide::Domain, BoundaryEnd::Right>(view);
  auto restored = repartition<MorphismSide::Codomain, BoundaryEnd::Right>(bent);
  static_assert(decltype(bent)::domain_type::size() == 2);
  static_assert(decltype(bent)::codomain_type::size() == 4);
  EXPECT_EQ(bent.codomain().template space<3>(), dual(a));
  EXPECT_EQ(restored.domain(), view.domain());
  EXPECT_EQ(restored.codomain(), view.codomain());

  // Also exercise the fixed-output path with complete storage.
  using Complete =
      BlockTensor<double, typename Result::domain_type, typename Result::codomain_type, PackedCompleteBlockStorage<>>;
  Complete fixed(sym, result.domain(), result.codomain());
  contract<3, 0>(fixed, left, right);
  for (std::size_t sector = 0; sector < 2; ++sector)
  {
    ResultKey const key{{sector, 0, 0, 0, sector, 0}};
    ResultKey const view_key{{0, 0, sector, sector, 0, 0}};
    ResultKey const bent_key{{0, 0, sector, 0, 0, sector}};
    auto original = result.block(key);
    auto lhs = left.block(Key{{sector, 0, 0, sector}});
    auto rhs = right.block(Key{{sector, 0, sector, 0}});
    ASSERT_TRUE(view.is_legal(view_key));
    ASSERT_TRUE(bent.is_legal(bent_key));
    EXPECT_EQ(view.block(view_key).mdspan().data_handle(), original.mdspan().data_handle());
    EXPECT_EQ(bent.block(bent_key).mdspan().data_handle(), original.mdspan().data_handle());
    EXPECT_EQ(restored.block(view_key).mdspan().data_handle(), original.mdspan().data_handle());
    EXPECT_NE(copied.block(view_key).mdspan().data_handle(), original.mdspan().data_handle());
    for_each_block_index(original, [&](auto i, auto j, auto l, auto k, auto m, auto n) {
      double expected = 0.0;
      for (index_type s = 0; s < lhs.extent(3); ++s)
        expected += lhs[i, j, k, s] * rhs[s, l, m, n];
      EXPECT_DOUBLE_EQ((original[i, j, l, k, m, n]), expected);
      EXPECT_DOUBLE_EQ((fixed.block(key)[i, j, l, k, m, n]), expected);
      EXPECT_DOUBLE_EQ((view.block(view_key)[l, j, i, m, n, k]), expected);
      EXPECT_DOUBLE_EQ((copied.block(view_key)[l, j, i, m, n, k]), expected);
      EXPECT_DOUBLE_EQ((bent.block(bent_key)[l, j, m, n, k, i]), expected);
      EXPECT_DOUBLE_EQ((restored.block(view_key)[l, j, i, m, n, k]), expected);
    });
  }
}

TYPED_TEST(HighOrderBlockTensorTest, GroupedContractionSeparatesOrderKeysAndDenseRank)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  BlockSpace const rows(sym, {{q0, 2}, {q1, 3}}, "rows");
  BlockSpace const bond(sym, {{q0, 2}, {q1, 1}}, "bond");
  BlockSpace const columns(sym, {{q0, 3}, {q1, 2}}, "columns");
  LocalSpace const local(sym, {q0, q0}, "repeated-local-charge");
  QNumSpace const fixed(q1, "fixed-charge");
  using Left =
      BlockTensor<double, Domain<BlockSpace, QNumSpace, Dual<QNumSpace>>, Codomain<LocalSpace, BlockSpace>, TypeParam>;
  using Right =
      BlockTensor<double, Domain<LocalSpace, BlockSpace>, Codomain<BlockSpace, QNumSpace, Dual<QNumSpace>>, TypeParam>;
  using Key = typename Left::key_type;
  Left left(sym, Domain{rows, fixed, dual(fixed)}, Codomain{local, bond},
            {Key{{0, 0, 0}}, Key{{0, 1, 0}}, Key{{1, 0, 1}}, Key{{1, 1, 1}}});
  Right right(sym, Domain{local, bond}, Codomain{columns, fixed, dual(fixed)},
              {Key{{0, 0, 0}}, Key{{1, 0, 0}}, Key{{0, 1, 1}}, Key{{1, 1, 1}}});
  static_assert(Left::order() == 5 && Left::key_coordinate_count() == 3 && Left::dense_block_order() == 2);
  for (std::size_t sector = 0; sector < 2; ++sector)
    for (std::size_t state = 0; state < 2; ++state)
    {
      auto lhs = left.block(Key{{sector, state, sector}});
      auto rhs = right.block(Key{{state, sector, sector}});
      for_each_block_index(lhs, [&](auto i, auto k) { lhs[i, k] = static_cast<double>(1 + i + 3 * k + state); });
      for_each_block_index(rhs, [&](auto k, auto j) { rhs[k, j] = static_cast<double>(2 + k + 5 * j + 2 * state); });
    }

  auto result = contract_adjacent<2>(left, right);
  using Result = decltype(result);
  static_assert(Result::order() == 6 && Result::key_coordinate_count() == 2 && Result::dense_block_order() == 2);
  ASSERT_EQ(result.stored_block_count(), 2);
  EXPECT_EQ(result.domain(), left.domain());
  EXPECT_EQ(result.codomain(), right.codomain());
  for (std::size_t sector = 0; sector < 2; ++sector)
  {
    auto output = result.block(typename Result::key_type{{sector, sector}});
    for_each_block_index(output, [&](auto i, auto j) {
      double expected = 0.0;
      for (std::size_t state = 0; state < 2; ++state)
      {
        auto lhs = left.block(Key{{sector, state, sector}});
        auto rhs = right.block(Key{{state, sector, sector}});
        for (index_type k = 0; k < lhs.extent(1); ++k)
          expected += lhs[i, k] * rhs[k, j];
      }
      EXPECT_DOUBLE_EQ((output[i, j]), expected);
    });
  }
}

template <class Orientation> class HighOrderBlockSvdTest : public ::testing::Test {};
using Orientations = ::testing::Types<std::false_type, std::true_type>;
TYPED_TEST_SUITE(HighOrderBlockSvdTest, Orientations);

TYPED_TEST(HighOrderBlockSvdTest, ReconstructsMultipleChargeFragmentsWithFiveLegFactors)
{
  constexpr bool four_domain = TypeParam::value;
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  BlockSpace const charged(sym, {{q0, 1}, {q1, 2}}, "charged");
  BlockSpace const neutral(sym, {{q0, 2}}, "neutral");
  auto domain = [&] {
    if constexpr (four_domain)
      return Domain{charged, charged, neutral, neutral};
    else
      return Domain{charged, neutral};
  }();
  auto codomain = [&] {
    if constexpr (four_domain)
      return Codomain{charged, neutral};
    else
      return Codomain{charged, charged, neutral, neutral};
  }();
  using Tensor = BlockTensor<double, decltype(domain), decltype(codomain), PackedCompleteBlockStorage<>>;
  Tensor tensor(sym, domain, codomain);
  ASSERT_EQ(tensor.stored_block_count(), 3);
  for (std::size_t ordinal = 0; ordinal < tensor.stored_block_count(); ++ordinal)
  {
    auto block = tensor.block_by_ordinal(ordinal);
    for_each_block_index(block, [&](auto a, auto b, auto c, auto d, auto e, auto f) {
      block[a, b, c, d, e, f] = static_cast<double>(1 + 2 * a + 3 * b + 5 * c + 7 * d + 11 * e + 13 * f +
                                                    (a + c + e) * (b + d + f) + 17 * ordinal);
    });
  }

  auto decomposition = block_svd(tensor);
  ASSERT_EQ(decomposition.sectors().size(), 2);
  auto factors = materialize_svd(decomposition, select_svd_states(decomposition.spectrum()));
  using Left = decltype(factors.left_singular_vectors);
  using Right = decltype(factors.right_singular_vectors_adjoint);
  static_assert(Left::order() == (four_domain ? 3 : 5));
  static_assert(Right::order() == (four_domain ? 5 : 3));
  EXPECT_EQ(factors.left_singular_vectors.codomain(), tensor.codomain());
  EXPECT_EQ(factors.right_singular_vectors_adjoint.domain(), tensor.domain());
  auto const& singular_bond = factors.singular_values.domain().template space<0>();
  ASSERT_EQ(singular_bond.size(), 2);
  for (auto const& sector : decomposition.sectors())
  {
    if (sector.charge == q1)
    {
      if constexpr (four_domain)
        EXPECT_EQ(sector.domain_fragments.size(), 2);
      else
        EXPECT_EQ(sector.codomain_fragments.size(), 2);
    }
    auto const found = std::ranges::find_if(singular_bond, [&](auto const& entry) { return entry.q == sector.charge; });
    ASSERT_NE(found, singular_bond.end());
    auto const bond = static_cast<std::size_t>(found - singular_bond.begin());
    auto values =
        factors.singular_values.diagonal_values(typename decltype(factors.singular_values)::key_type{{bond, bond}});
    for (auto const& key : sector.stored_source_keys)
    {
      auto original = tensor.block(key);
      auto check = [&](auto left_key, auto right_key) {
        ASSERT_TRUE(factors.left_singular_vectors.is_legal(left_key));
        ASSERT_TRUE(factors.right_singular_vectors_adjoint.is_legal(right_key));
        auto lhs = factors.left_singular_vectors.block(left_key);
        auto rhs = factors.right_singular_vectors_adjoint.block(right_key);
        for_each_block_index(original, [&](auto a, auto b, auto c, auto d, auto e, auto f) {
          double reconstructed = 0.0;
          for (index_type s = 0; s < static_cast<index_type>(values.size()); ++s)
          {
            if constexpr (four_domain)
              reconstructed += lhs[s, e, f] * values[s] * rhs[a, b, c, d, s];
            else
              reconstructed += lhs[s, c, d, e, f] * values[s] * rhs[a, b, s];
          }
          EXPECT_NEAR(reconstructed, (original[a, b, c, d, e, f]), 1.0e-11);
        });
      };
      if constexpr (four_domain)
        check(typename Left::key_type{{bond, key.coordinate(4), key.coordinate(5)}},
              typename Right::key_type{
                  {key.coordinate(0), key.coordinate(1), key.coordinate(2), key.coordinate(3), bond}});
      else
        check(
            typename Left::key_type{{bond, key.coordinate(2), key.coordinate(3), key.coordinate(4), key.coordinate(5)}},
            typename Right::key_type{{key.coordinate(0), key.coordinate(1), bond}});
    }
  }
}

TEST(HighOrderBlockTensorTest, ConstructsOrderNineTensorWithChargedFixedFactor)
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  BlockSpace const leg(sym, {{q0, 1}, {q1, 1}}, "leg");
  QNumSpace const charge(q1, "charge");
  using Tensor =
      BlockTensor<double, Domain<BlockSpace, BlockSpace, BlockSpace, BlockSpace>,
                  Codomain<BlockSpace, BlockSpace, BlockSpace, BlockSpace, QNumSpace>, PackedSparseBlockStorage<>>;
  using Key = Tensor::key_type;
  Key const key{{1, 0, 0, 0, 0, 0, 0, 0}};
  Tensor tensor(sym, Domain{leg, leg, leg, leg}, Codomain{leg, leg, leg, leg, charge}, {key});
  static_assert(Tensor::order() == 9 && Tensor::key_coordinate_count() == 8 && Tensor::dense_block_order() == 8);
  EXPECT_FALSE(tensor.is_legal(Key{}));
  tensor.block(key)[0, 0, 0, 0, 0, 0, 0, 0] = 42.0;
  EXPECT_DOUBLE_EQ((tensor.block(key)[0, 0, 0, 0, 0, 0, 0, 0]), 42.0);
}

TEST(HighOrderBlockTensorTest, AsyncContractionRetainsMatrixGeometryAtHigherTensorOrder)
{
  Symmetry const sym{"N:U(1)"};
  auto const q1 = make_qnum(sym, {{"N", 1}});
  BlockSpace const leg(sym, {{q1, 2}}, "leg");
  QNumSpace const fixed(q1, "fixed");
  using Storage = AsyncSeparateSparseBlockStorage<>;
  using Left = BlockTensor<double, Domain<BlockSpace, QNumSpace, Dual<QNumSpace>>, Codomain<BlockSpace>, Storage>;
  using Right = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace, QNumSpace, Dual<QNumSpace>>, Storage>;
  Left::key_type const key{{0, 0}};
  Left left(sym, Domain{leg, fixed, dual(fixed)}, Codomain{leg}, {key});
  Right right(sym, Domain{leg}, Codomain{leg, fixed, dual(fixed)}, {key});
  // Initialize before any epochs are enrolled, as in the ordinary async matrix tests.
  auto& lhs = left.async_block(key).unsafe_value_ref();
  auto& rhs = right.async_block(key).unsafe_value_ref();
  lhs[0, 0] = 1.0;
  lhs[1, 1] = 2.0;
  rhs[0, 0] = 3.0;
  rhs[1, 1] = 4.0;

  async::DebugScheduler scheduler;
  async::ScopedScheduler scoped(&scheduler);
  auto result = contract<3, 0>(left, right);
  static_assert(decltype(result)::order() == 6 && decltype(result)::dense_block_order() == 2);
  static_assert(std::same_as<typename decltype(result)::storage_policy, Storage>);
  EXPECT_EQ(result.domain(), left.domain());
  EXPECT_EQ(result.codomain(), right.codomain());
  auto const& block = result.async_block(key).get_wait(scheduler);
  EXPECT_DOUBLE_EQ((block[0, 0]), 3.0);
  EXPECT_DOUBLE_EQ((block[1, 1]), 8.0);
  EXPECT_DOUBLE_EQ((block[0, 1]), 0.0);
  EXPECT_DOUBLE_EQ((block[1, 0]), 0.0);
}
} // namespace
