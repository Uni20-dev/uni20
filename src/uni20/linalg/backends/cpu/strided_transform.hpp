#pragma once

/**
 * \file strided_transform.hpp
 * \ingroup internal
 * \brief CPU executor for compact strided elementwise iteration plans.
 */

#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/iteration_plan.hpp>

#include <array>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::linalg::cpu::detail
{

template <bool ReadsOutput, class Output, class Operation, class... Inputs> consteval bool strided_transform_operation()
{
  using output_type = std::remove_cvref_t<Output>;
  if constexpr (!(MutableMdspanLike<Output> && AffineMdspanLike<Output>) || (!(AffineMdspanLike<Inputs> && ...)) ||
                !((output_type::rank() == std::remove_cvref_t<Inputs>::rank()) && ...))
  {
    return false;
  }
  else if constexpr (ReadsOutput)
  {
    return requires(Operation& operation) {
      std::declval<typename output_type::reference>() =
          std::invoke(operation, std::declval<typename output_type::reference>(),
                      std::declval<typename std::remove_cvref_t<Inputs>::reference>()...);
    };
  }
  else
  {
    return requires(Operation& operation) {
      std::declval<typename output_type::reference>() =
          std::invoke(operation, std::declval<typename std::remove_cvref_t<Inputs>::reference>()...);
    };
  }
}

template <bool ReadsOutput, class Operation, AffineMdspanLike Output, AffineMdspanLike... Inputs>
  requires MutableMdspanLike<Output>
class strided_transform_executor {
  public:
    using operation_type = std::decay_t<Operation>;
    static constexpr std::size_t input_count = sizeof...(Inputs);
    static constexpr std::size_t span_count = 1 + input_count;
    using index_type = std::ptrdiff_t;
    using offset_type = std::array<index_type, span_count>;
    using plan_entry = extent_strides<span_count>;

    template <class FwdOperation>
    strided_transform_executor(FwdOperation&& operation, Output const& output, Inputs const&... inputs)
        : handles_{output.data_handle(), inputs.data_handle()...}, accessors_{output.accessor(), inputs.accessor()...},
          operation_(std::forward<FwdOperation>(operation))
    {}

    template <class Plan> void run(Plan const& plan, offset_type offsets)
    {
      switch (plan.size())
      {
        case 0:
          this->run_unrolled(offsets, plan.data(), std::integral_constant<std::size_t, 0>{});
          return;
        case 1:
          this->run_unrolled(offsets, plan.data(), std::integral_constant<std::size_t, 1>{});
          return;
        case 2:
          this->run_unrolled(offsets, plan.data(), std::integral_constant<std::size_t, 2>{});
          return;
        case 3:
          this->run_unrolled(offsets, plan.data(), std::integral_constant<std::size_t, 3>{});
          return;
        default:
          this->run_dynamic(offsets, plan.data(), plan.size());
          return;
      }
    }

  private:
    template <std::size_t Operand> decltype(auto) access(std::ptrdiff_t offset)
    {
      using span_type = std::tuple_element_t<Operand, std::tuple<Output, Inputs...>>;
      using mapping_offset_type =
          decltype(uni20::detail::mapping_origin(std::declval<typename span_type::mapping_type const&>()));
      // The plan visits reachable mapping offsets. Convert as mdspan indexing does.
      return std::as_const(std::get<Operand>(accessors_))
          .access(std::get<Operand>(handles_),
                  static_cast<typename span_type::offset_type>(static_cast<mapping_offset_type>(offset)));
    }

    void apply(offset_type const& offsets)
    {
      [&]<std::size_t... Input>(std::index_sequence<Input...>) {
        auto&& output = this->access<0>(offsets[0]);
        if constexpr (ReadsOutput)
        {
          std::forward<decltype(output)>(output) =
              std::invoke(operation_, std::forward<decltype(output)>(output),
                          this->access<Input + 1>(offsets[Input + 1])...);
        }
        else
        {
          std::forward<decltype(output)>(output) =
              std::invoke(operation_, this->access<Input + 1>(offsets[Input + 1])...);
        }
      }(std::make_index_sequence<input_count>{});
    }

    template <std::size_t Depth>
    void run_unrolled(offset_type offsets, plan_entry const* plan, std::integral_constant<std::size_t, Depth>)
    {
      if constexpr (Depth == 0)
      {
        this->apply(offsets);
      }
      else
      {
        index_type const extent = plan->extent;
        for (index_type index = 0; index < extent; ++index)
        {
          this->run_unrolled(offsets, plan + 1, std::integral_constant<std::size_t, Depth - 1>{});
          for (std::size_t span = 0; span < span_count; ++span)
            offsets[span] += plan->strides[span];
        }
      }
    }

    void run_dynamic(offset_type offsets, plan_entry const* plan, std::size_t depth)
    {
      constexpr std::size_t unroll_depth = 3;
      index_type const extent = plan->extent;
      for (index_type index = 0; index < extent; ++index)
      {
        if (depth == unroll_depth + 1)
          this->run_unrolled(offsets, plan + 1, std::integral_constant<std::size_t, 3>{});
        else
          this->run_dynamic(offsets, plan + 1, depth - 1);

        for (std::size_t span = 0; span < span_count; ++span)
          offsets[span] += plan->strides[span];
      }
    }

    std::tuple<typename Output::data_handle_type, typename Inputs::data_handle_type...> handles_;
    std::tuple<typename Output::accessor_type, typename Inputs::accessor_type...> accessors_;
    [[no_unique_address]] operation_type operation_;
};

template <bool ReadsOutput, class Output, class Operation, class... Inputs>
  requires(strided_transform_operation<ReadsOutput, Output, Operation, Inputs...>())
void transform_strided(Output output, Operation&& operation, Inputs const&... inputs)
{
  auto mappings = std::tuple{output.mapping(), inputs.mapping()...};
  auto plan = make_multi_iteration_plan(mappings);
  CHECK(plan.representable, "elementwise iteration plan exceeds host index range");

  strided_transform_executor<ReadsOutput, Operation, Output, Inputs...> executor{std::forward<Operation>(operation),
                                                                                 output, inputs...};
  executor.run(plan.dimensions, plan.base_offsets);
}

} // namespace uni20::linalg::cpu::detail
