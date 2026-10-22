#include <uni20/krylov/block_tensor_vector.hpp>
#include <uni20/krylov/symmetric_lanczos.hpp>
#include <uni20/krylov/gmres.hpp>
#include <uni20/krylov/gmres_refinement.hpp>
#include <uni20/symmetry/block_space.hpp>
#include <uni20/symmetry/block_tensor.hpp>
#include <uni20/symmetry/block_tensor_contract.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace
{

using Storage = uni20::SeparateSparseBlockStorage<>;
using Tensor =
    uni20::BlockTensor<double, uni20::Domain<uni20::BlockSpace>, uni20::Codomain<uni20::BlockSpace>, Storage>;
using Key = typename Tensor::key_type;

struct U1BlockDiagonalMatvec
{
    Tensor const* matrix;

    void operator()(Tensor& output, Tensor const& input) const { uni20::contract<1, 0>(output, *matrix, input); }
};

using U1BlockDiagonalOps = uni20::krylov::BlockTensorMatrixFreeOps<Tensor, U1BlockDiagonalMatvec>;

static_assert(uni20::krylov::KrylovVectorOps<uni20::krylov::BlockTensorVectorOps<Tensor>, Tensor, double>);
static_assert(!uni20::krylov::KrylovOperator<uni20::krylov::BlockTensorVectorOps<Tensor>, Tensor>);
static_assert(uni20::krylov::KrylovMatrixFreeOperator<U1BlockDiagonalOps, Tensor, double>);

template <class Source>
concept CanBindBlockSpace = requires(uni20::krylov::BlockTensorVectorOps<Tensor>& ops, Source&& source) {
  uni20::krylov::krylov_space<double>(ops, std::forward<Source>(source));
};
static_assert(CanBindBlockSpace<Tensor&> && CanBindBlockSpace<Tensor const&>);
static_assert(!CanBindBlockSpace<Tensor> && !CanBindBlockSpace<Tensor const>);
static_assert(CanBindBlockSpace<decltype(uni20::as_block_tensor_view(std::declval<Tensor&>()))>);

auto make_prototype(uni20::Symmetry const& symmetry, uni20::BlockSpace const& space) -> Tensor
{
  return Tensor(symmetry, uni20::Domain{space}, uni20::Codomain{space}, {Key{{0, 0}}, Key{{1, 1}}});
}

void fill(Tensor& tensor, double value)
{
  for (std::size_t ordinal = 0; ordinal < tensor.stored_block_count(); ++ordinal)
  {
    auto block = tensor.block_by_ordinal(ordinal);
    for (uni20::index_type column = 0; column < block.extent(1); ++column)
      for (uni20::index_type row = 0; row < block.extent(0); ++row)
        block[row, column] = value;
  }
}

void fill_diagonal_operator(Tensor& matrix)
{
  uni20::set_zero(matrix);
  auto sector_zero = matrix.block(Key{{0, 0}});
  sector_zero[0, 0] = 1.0;
  sector_zero[1, 1] = 3.0;
  matrix.block(Key{{1, 1}})[0, 0] = 2.0;
}

TEST(BlockTensorVectorOpsTest, FreezesStructureAndAllocatesWithoutDenseProjection)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const space(symmetry, {{q0, 2}, {q1, 1}}, "state");
  Tensor prototype = make_prototype(symmetry, space);
  fill(prototype, 2.0);

  uni20::krylov::BlockTensorVectorOps<Tensor> ops(prototype);
  EXPECT_EQ(ops.problem_dimension(), 5);
  EXPECT_EQ(ops.vector_dimension(prototype), 5);

  Tensor allocated = ops.allocate_like(prototype);
  EXPECT_EQ(allocated.symmetry(), prototype.symmetry());
  EXPECT_EQ(allocated.domain(), prototype.domain());
  EXPECT_EQ(allocated.codomain(), prototype.codomain());
  EXPECT_TRUE(std::ranges::equal(allocated.stored_keys(), prototype.stored_keys()));
  ops.set_zero(allocated);
  EXPECT_DOUBLE_EQ(ops.norm(allocated), 0.0);

  auto relabelled_space = space;
  relabelled_space.set_label("other-state");
  Tensor relabelled = make_prototype(symmetry, relabelled_space);
  EXPECT_THROW(static_cast<void>(ops.vector_dimension(relabelled)), std::invalid_argument);

  Tensor different_pattern(symmetry, uni20::Domain{space}, uni20::Codomain{space}, {Key{{0, 0}}});
  EXPECT_THROW(static_cast<void>(ops.vector_dimension(different_pattern)), std::invalid_argument);
}

TEST(BlockTensorVectorOpsTest, CountsCompressedDiagonalDegreesOfFreedom)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const space(symmetry, {{q0, 3}, {q1, 2}}, "diagonal-state");
  using DiagonalTensor = uni20::BlockTensor<double, uni20::Domain<uni20::BlockSpace>,
                                            uni20::Codomain<uni20::BlockSpace>, uni20::PackedDiagonalBlockStorage<>>;
  using DiagonalKey = typename DiagonalTensor::key_type;
  DiagonalTensor prototype(symmetry, uni20::Domain{space}, uni20::Codomain{space},
                           {DiagonalKey{{0, 0}}, DiagonalKey{{1, 1}}});

  uni20::krylov::BlockTensorVectorOps<DiagonalTensor> ops(prototype);
  EXPECT_EQ(ops.problem_dimension(), 5);
  EXPECT_EQ(ops.vector_dimension(prototype), 5);
  auto vector_space = uni20::krylov::krylov_space<double>(ops, prototype);
  static_assert(uni20::DiagonalBlockTensorView<typename decltype(vector_space)::prototype_type>);
  auto allocated = ops.allocate_like(vector_space.prototype);
  EXPECT_EQ(ops.vector_dimension(allocated), 5);
  EXPECT_TRUE(std::ranges::equal(allocated.storage().offsets(), prototype.storage().offsets()));
  ops.set_zero(allocated);
  EXPECT_EQ(ops.norm(allocated), 0.0);
}

TEST(BlockTensorVectorOpsTest, AllocatesCompletePackedVectorsFromTheirBoundaries)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const space(symmetry, {{q0, 2}, {q1, 1}}, "state");
  using CompleteTensor = uni20::BlockTensor<double, uni20::Domain<uni20::BlockSpace>,
                                            uni20::Codomain<uni20::BlockSpace>, uni20::PackedCompleteBlockStorage<>>;
  CompleteTensor prototype(symmetry, uni20::Domain{space}, uni20::Codomain{space});
  uni20::krylov::BlockTensorVectorOps<CompleteTensor> ops(prototype);
  prototype.block_by_ordinal(0)[0, 0] = 7.0;

  auto vector_space = uni20::krylov::krylov_space<double>(ops, uni20::as_block_tensor_view(prototype));
  auto allocated = ops.allocate_like(vector_space.prototype);
  EXPECT_TRUE(allocated.has_all_legal_blocks());
  EXPECT_TRUE(std::ranges::equal(allocated.stored_keys(), prototype.stored_keys()));
  EXPECT_TRUE(std::ranges::equal(allocated.storage().offsets(), prototype.storage().offsets()));
  EXPECT_NE(allocated.storage().buffer().data(), prototype.storage().buffer().data());
  ops.set_zero(allocated);
  EXPECT_DOUBLE_EQ(ops.norm(allocated), 0.0);
}

// A context-requiring host policy exercises placement retention without CUDA.
// No default allocation path exists, so losing the context cannot silently pass.
struct AllocationContext
{
    std::size_t allocations = 0;
};
template <class T> struct ContextBuffer : uni20::HostBuffer<T>
{
    AllocationContext* context;
    ContextBuffer(AllocationContext& ctx, std::size_t size, uni20::StorageInitialization initialization)
        : uni20::HostBuffer<T>(size, initialization), context(&ctx)
    {
      ++ctx.allocations;
    }
};
struct ContextHostStorage : uni20::HostStorage
{
    template <class T> using storage_t = ContextBuffer<T>;
    template <class T>
    static auto make_packed_storage(AllocationContext& ctx, std::size_t size, std::span<std::size_t const>,
                                    uni20::StorageInitialization initialization) -> storage_t<T>
    {
      return {ctx, size, initialization};
    }
    template <class T> static auto make_handle(storage_t<T>& buffer) -> T* { return buffer.data(); }
    template <class T> static auto make_handle(storage_t<T> const& buffer) -> T const* { return buffer.data(); }
    template <class T> static auto allocation_context(storage_t<T> const& buffer) -> AllocationContext&
    {
      return *buffer.context;
    }
};

template <class StoragePolicy> class BlockPrototypeContextTest : public ::testing::Test {};
using ContextStorageTypes = ::testing::Types<uni20::PackedSparseBlockStorage<ContextHostStorage, 64>,
                                            uni20::PackedCompleteBlockStorage<ContextHostStorage, 64>>;
TYPED_TEST_SUITE(BlockPrototypeContextTest, ContextStorageTypes);

TYPED_TEST(BlockPrototypeContextTest, RetainsContextThroughTemporaryViewsIncludingEmptySpaces)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  using T = uni20::BlockTensor<double, uni20::Domain<uni20::BlockSpace>, uni20::Codomain<uni20::BlockSpace>, TypeParam>;
  for (bool empty : {false, true})
  {
    auto const q0 = uni20::QNum::identity(symmetry);
    uni20::BlockSpace const space = empty ? uni20::BlockSpace(symmetry, {})
                                          : uni20::BlockSpace(symmetry, {{q0, 2}});
    AllocationContext allocation;
    auto prototype = [&] {
      if constexpr (uni20::CompleteBlockStorage<TypeParam>)
        return T(symmetry, uni20::Domain{space}, uni20::Codomain{space}, allocation);
      else
      {
        std::vector<typename T::key_type> keys;
        if (!empty) keys.push_back(typename T::key_type{{0, 0}});
        return T(symmetry, uni20::Domain{space}, uni20::Codomain{space}, keys, allocation);
      }
    }();
    uni20::krylov::BlockTensorVectorOps<T> ops(prototype);
    auto retained = [&] {
      auto temporary = uni20::as_block_tensor_view(prototype);
      return uni20::krylov::krylov_space<double>(ops, temporary);
    }();
    EXPECT_EQ(allocation.allocations, 1);
    EXPECT_EQ(&retained.prototype.allocation_context(), &allocation);
    auto workspace = ops.allocate_like(retained.prototype);
    EXPECT_EQ(allocation.allocations, 2);
    EXPECT_EQ(&workspace.allocation_context(), &allocation);
    EXPECT_TRUE(std::ranges::equal(workspace.storage().offsets(), prototype.storage().offsets()));
    EXPECT_TRUE(std::ranges::equal(workspace.stored_keys(), prototype.stored_keys()));
    EXPECT_EQ(workspace.domain(), prototype.domain());
    EXPECT_EQ(workspace.codomain(), prototype.codomain());
    if (!empty)
    {
      EXPECT_NE(workspace.storage().buffer().data(), prototype.storage().buffer().data());
    }
  }
}

TEST(BlockTensorVectorOpsTest, RunsU1BlockTensorThroughSymmetricLanczos)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const space(symmetry, {{q0, 2}, {q1, 1}}, "state");
  Tensor initial = make_prototype(symmetry, space);
  fill(initial, 1.0);
  Tensor matrix = make_prototype(symmetry, space);
  fill_diagonal_operator(matrix);
  U1BlockDiagonalOps ops(initial, U1BlockDiagonalMatvec{.matrix = &matrix});
  Tensor wrong_output(symmetry, uni20::Domain{space}, uni20::Codomain{space}, {Key{{0, 0}}});
  EXPECT_THROW(ops.matvec(wrong_output, initial), std::invalid_argument);

  uni20::krylov::SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 5;
  params.spectrum = uni20::krylov::SpectrumPart::SmallestAlgebraic;
  params.compute_eigenvectors = true;

  auto result = uni20::krylov::symmetric_lanczos_standard<double>(ops, initial, params);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.size(), 2);
  EXPECT_NEAR(result.eigenvalues[0], 1.0, 1.0e-12);
  EXPECT_NEAR(result.eigenvalues[1], 2.0, 1.0e-12);
  EXPECT_EQ(result.converged_count, 2);
  for (std::size_t index = 0; index < result.eigenvectors.size(); ++index)
  {
    Tensor residual = ops.allocate_like(result.eigenvectors[index]);
    ops.matvec(residual, result.eigenvectors[index]);
    ops.axpy(residual, -result.eigenvalues[index], result.eigenvectors[index]);
    EXPECT_LT(ops.norm(residual), 1.0e-11);
    EXPECT_NEAR(ops.norm(result.eigenvectors[index]), 1.0, 1.0e-12);
    EXPECT_TRUE(std::ranges::equal(result.eigenvectors[index].stored_keys(), initial.stored_keys()));
  }
}

TEST(BlockTensorVectorOpsTest, GmresPreservesU1BlocksAndBoundaries)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const space(symmetry, {{q0, 2}, {q1, 1}}, "state");
  Tensor x = make_prototype(symmetry, space);
  Tensor rhs = make_prototype(symmetry, space);
  Tensor matrix = make_prototype(symmetry, space);
  fill(x, 0.0);
  fill(rhs, 1.0);
  fill_diagonal_operator(matrix);
  U1BlockDiagonalOps ops(x, U1BlockDiagonalMatvec{.matrix = &matrix});
  auto info = uni20::krylov::gmres_with_info<double>(ops, ops, x, rhs, {.relative_tolerance = 1e-13});
  ASSERT_TRUE(info.succeeded());
  EXPECT_EQ(x.symmetry(), rhs.symmetry());
  EXPECT_EQ(x.domain(), rhs.domain());
  EXPECT_EQ(x.codomain(), rhs.codomain());
  EXPECT_TRUE(std::ranges::equal(x.stored_keys(), rhs.stored_keys()));
  auto first = x.block(Key{{0, 0}});
  EXPECT_NEAR((first[0, 0]), 1.0, 1e-13);
  EXPECT_NEAR((first[0, 1]), 1.0, 1e-13);
  EXPECT_NEAR((first[1, 0]), 1.0 / 3.0, 1e-13);
  EXPECT_NEAR((first[1, 1]), 1.0 / 3.0, 1e-13);
  EXPECT_NEAR((x.block(Key{{1, 1}})[0, 0]), 0.5, 1e-13);

  fill(x, 0.0);
  auto transfer = [](Tensor& out, Tensor const& in, uni20::Precision p) {
    uni20::require_precision<double>(p);
    uni20::copy(out, in);
  };
  uni20::krylov::IdentityPreconditioner preconditioner{ops};
  auto working_space = [&] {
    auto temporary_view = uni20::as_block_tensor_view(std::as_const(x));
    return uni20::krylov::krylov_space<double>(ops, temporary_view);
  }();
  static_assert(uni20::BorrowedBlockTensorView<typename decltype(working_space)::prototype_type>);
  static_assert(std::same_as<typename decltype(working_space)::vector_type, Tensor>);
  EXPECT_EQ(working_space.prototype.block(Key{{0, 0}}).mdspan().data_handle(),
            x.block(Key{{0, 0}}).mdspan().data_handle());
  uni20::krylov::GmresRefinementContext context{.working = std::move(working_space),
                                                .inner = uni20::krylov::krylov_space<double>(ops, rhs),
                                                .residual = uni20::krylov::krylov_space<double>(
                                                    ops, uni20::as_block_tensor_view(rhs)),
                                                .inner_operator = ops,
                                                .residual_operator = ops,
                                                .preconditioner = preconditioner,
                                                .transfer = transfer};
  auto refined = uni20::krylov::gmres_refine_with_info(context, x, rhs);
  ASSERT_TRUE(refined.succeeded());
  EXPECT_EQ(x.symmetry(), rhs.symmetry());
  EXPECT_EQ(x.domain(), rhs.domain());
  EXPECT_EQ(x.codomain(), rhs.codomain());
  EXPECT_TRUE(std::ranges::equal(x.stored_keys(), rhs.stored_keys()));
  EXPECT_NEAR((x.block(Key{{1, 1}})[0, 0]), 0.5, 1e-14);
}

} // namespace
