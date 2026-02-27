// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

#define OLD 1
namespace ck_tile {
// Default policy for GemmPipelineAgBgCrCompAsync
// Customized methods: MakeALdsBlockDescriptor, MakeBLdsBlockDescriptor
// GetBlockGemm implementation is copied from GemmPipelineAgBgCrCompV4DefaultPolicy
struct GemmPipelineAgBgCrCompAsyncDefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompAsyncDefaultPolicy>
{
    static constexpr auto ATileAccessPattern = tile_distribution_pattern::warp_raked;
    static constexpr auto BTileAccessPattern = tile_distribution_pattern::warp_raked;

    static constexpr index_t kLdsRowBytes = 64 * 4;
    static constexpr index_t DWORDx4      = 16;
    static constexpr index_t kLdsBanks          = 64; // This policy is only for gfx950 for now
    static constexpr index_t kLdsBankBytes      = 4;
    static constexpr index_t kBytesPerLdsRow    = kLdsBanks * kLdsBankBytes;
    static constexpr index_t kMaxVecWidth       = get_max_mem_vec_inst_width();
    static constexpr index_t kVecLoadsPerLdsRow = kBytesPerLdsRow / kMaxVecWidth;
    static_assert(kMaxVecWidth == 16);
    static_assert(kVecLoadsPerLdsRow == 16);

    template <typename Problem,
              typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
    CK_TILE_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
#if !OLD
        using ADataType             = remove_cvref_t<typename Problem::ADataType>;
#endif
        if constexpr(is_a_load_tr<Problem>)
        {
            // TODO: better LDS descriptor for performance
            // This branch is reusing the logic from
            // UniversalGemmBasePolicy::MakeALdsBlockDescriptor
            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor( //
                make_tuple(number<KPerBlock>{}, number<MPerBlock>{}),
                make_tuple(number<MPerBlock>{}, number<1>{}),
                number<MPerBlock>{},
                number<1>{});
            return a_lds_block_desc_0;
        }
        else
        {
            constexpr index_t KPack = GetSmemPackA<Problem>(); // k elements per thread 8
#if !OLD
            constexpr index_t KPacksPerXorShuffle =
                ck_tile::max(kBytesPerLdsRow / static_cast<index_t>(sizeof(ADataType)), KPerBlock) /
                KPack; // 16
#endif

            constexpr index_t L3 = KPack;
#if !OLD 
            constexpr index_t L2 = KPacksPerXorShuffle;
            constexpr index_t L1 = ck_tile::min(
                kVecLoadsPerLdsRow, integer_divide_ceil(MPerBlock * KPerBlock, L2 * L3)); // 16
            constexpr index_t L0 = integer_divide_ceil(MPerBlock * KPerBlock, L1 * L2 * L3);
            static_assert(L3 == 8);
            static_assert(L2 == 16);
            static_assert(L1 == 16);
            static_assert(L0 == 4);
#else
            constexpr index_t L2 = 64 * 2 / KPack;
            constexpr index_t L1 = 4;
            constexpr index_t L0 = MPerBlock * KPerBlock / (L1 * L2 * L3);
#endif
            constexpr auto a_lds_block_desc_0 =
                make_naive_tensor_descriptor(make_tuple(L0, L1, L2, L3),
                                             make_tuple(L1 * L2 * L3, L2 * L3, L3, 1),
                                             number<KPack>{},
                                             number<1>{});

            const auto a_lds_block_desc_1 = transform_tensor_descriptor(
                a_lds_block_desc_0,
                make_tuple(make_pass_through_transform(L0),
                           make_xor_transform(make_tuple(number<L1>{}, number<L2>{})),
                           make_pass_through_transform(L3)),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

#if !OLD
            const auto a_lds_block_desc_2 = transform_tensor_descriptor(
                a_lds_block_desc_1,
                make_tuple(make_merge_transform(make_tuple(L0, L1, L2, L3))),
                make_tuple(sequence<0, 1, 2, 3>{}),
                make_tuple(sequence<0>{}));
#else
            constexpr index_t KPacksPerBlock = KPerBlock / KPack;
            constexpr index_t MRowsPerLdsRow = 64 * 2 / KPerBlock;
            const auto a_lds_block_desc_2    = transform_tensor_descriptor(
                a_lds_block_desc_1,
                make_tuple(make_pass_through_transform(L0),
                           make_pass_through_transform(L1),
                           make_unmerge_transform(make_tuple(MRowsPerLdsRow, KPacksPerBlock)),
                           make_pass_through_transform(L3)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}));
#endif

#if !OLD
            const auto a_lds_block_desc_3 = transform_tensor_descriptor(
                a_lds_block_desc_2,
                make_tuple(make_unmerge_transform(make_tuple(MPerBlock, KPerBlock))),
                make_tuple(sequence<0>{}),
                make_tuple(sequence<0, 1>{}));
#else
            const auto a_lds_block_desc_3 = transform_tensor_descriptor(
                a_lds_block_desc_2,
                make_tuple(make_merge_transform(make_tuple(L0, L1, MRowsPerLdsRow)),
                           make_merge_transform(make_tuple(KPacksPerBlock, L3))),
                make_tuple(sequence<0, 1, 2>{}, sequence<3, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
#endif
            return a_lds_block_desc_3;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
#if !OLD
        using BDataType             = remove_cvref_t<typename Problem::BDataType>;
#endif
        if constexpr(is_b_load_tr<Problem>)
        {
            // TODO: better LDS descriptor for performance
            // This branch is reusing the logic from
            // UniversalGemmBasePolicy::MakeBLdsBlockDescriptor
            constexpr auto b_lds_block_desc_0 =
                make_naive_tensor_descriptor(make_tuple(number<KPerBlock>{}, number<NPerBlock>{}),
                                             make_tuple(number<NPerBlock>{}, number<1>{}),
                                             number<NPerBlock>{},
                                             number<1>{});
            return b_lds_block_desc_0;
        }
        else
        {
            constexpr index_t KPack = GetSmemPackB<Problem>();
#if !OLD
            constexpr index_t KPacksPerXorShuffle =
                ck_tile::max(kBytesPerLdsRow / static_cast<index_t>(sizeof(BDataType)), KPerBlock) /
                KPack;
            constexpr index_t L3 = KPack;
            constexpr index_t L2 = KPacksPerXorShuffle;
            constexpr index_t L1 = ck_tile::min(
                kVecLoadsPerLdsRow, integer_divide_ceil(NPerBlock * KPerBlock, L2 * L3));
            constexpr index_t L0 = integer_divide_ceil(NPerBlock * KPerBlock, L1 * L2 * L3);
#else
            constexpr index_t L3 = KPack;
            constexpr index_t L2 = 64 * 2 / KPack;
            constexpr index_t L1 = 4;
            constexpr index_t L0 = NPerBlock * KPerBlock / (L1 * L2 * L3);
#endif
            constexpr auto b_lds_block_desc_0 =
                make_naive_tensor_descriptor(make_tuple(L0, L1, L2, L3),
                                             make_tuple(L1 * L2 * L3, L2 * L3, L3, 1),
                                             number<KPack>{},
                                             number<1>{});

            const auto b_lds_block_desc_1 = transform_tensor_descriptor(
                b_lds_block_desc_0,
                make_tuple(make_pass_through_transform(L0),
                           make_xor_transform(make_tuple(number<L1>{}, number<L2>{})),
                           make_pass_through_transform(L3)),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

#if !OLD
            const auto b_lds_block_desc_2 = transform_tensor_descriptor(
                b_lds_block_desc_1,
                make_tuple(make_merge_transform(make_tuple(L0, L1, L2, L3))),
                make_tuple(sequence<0, 1, 2, 3>{}),
                make_tuple(sequence<0>{}));

            const auto b_lds_block_desc_3 = transform_tensor_descriptor(
                b_lds_block_desc_2,
                make_tuple(make_unmerge_transform(make_tuple(NPerBlock, KPerBlock))),
                make_tuple(sequence<0>{}),
                make_tuple(sequence<0, 1>{}));
#else
            constexpr index_t KPacksPerBlock = KPerBlock / KPack;
            constexpr index_t NRowsPerLdsRow = 64 * 2 / KPerBlock;
            const auto b_lds_block_desc_2    = transform_tensor_descriptor(
                b_lds_block_desc_1,
                make_tuple(make_pass_through_transform(L0),
                           make_pass_through_transform(L1),
                           make_unmerge_transform(make_tuple(NRowsPerLdsRow, KPacksPerBlock)),
                           make_pass_through_transform(L3)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}));

            const auto b_lds_block_desc_3 = transform_tensor_descriptor(
                b_lds_block_desc_2,
                make_tuple(make_merge_transform(make_tuple(L0, L1, NRowsPerLdsRow)),
                           make_merge_transform(make_tuple(KPacksPerBlock, L3))),
                make_tuple(sequence<0, 1, 2>{}, sequence<3, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
#endif
            return b_lds_block_desc_3;
        }
    }

    // Methods for async byte-based loading (similar to mx_flatmm)
    template <typename Problem, typename WindowTmp>
    CK_TILE_DEVICE static constexpr auto MakeAAsyncLoadBytesDramWindow(const WindowTmp& window_tmp)
    {
        using ADataType               = remove_cvref_t<typename Problem::ADataType>;
        constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
        constexpr index_t MPerBlock   = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;

        constexpr auto ndims = std::decay_t<decltype(window_tmp)>::get_num_of_dimension();
        static_assert(ndims == 2, "only support 2D tensor");
        auto&& tensor_view_tmp  = window_tmp.get_bottom_tensor_view();
        const auto [rows, cols] = tensor_view_tmp.get_tensor_descriptor().get_lengths();

#if OLD
        constexpr index_t kElementsPerLoad = kMaxVecWidth / (sizeof(ADataType) * APackedSize);
        constexpr index_t kLoadsPerBlockK  = KPerBlock / kElementsPerLoad;
        static_assert(kMaxVecWidth > sizeof(ADataType) * APackedSize);
        static_assert(KPerBlock >= kElementsPerLoad);
        constexpr index_t K2 = kElementsPerLoad;
        constexpr index_t K1 = kLoadsPerBlockK;
        const index_t K0     = integer_divide_ceil(cols, KPerBlock);
#else
        constexpr index_t K2 = DWORDx4 / (sizeof(ADataType) * APackedSize);
        constexpr index_t K1 = KPerBlock / K2;
        const index_t K0     = cols / (K1 * K2);
#endif
        const auto col_lens  = make_tuple(K0, number<K1>{}, number<K2>{});

#if OLD
        constexpr index_t M2 = integer_divide_ceil(kVecLoadsPerLdsRow, kLoadsPerBlockK);
        // const index_t M1     = 16;
        const index_t M1     = ck_tile::min(kVecLoadsPerLdsRow, integer_divide_ceil(rows, M2));
        const index_t M0     = integer_divide_ceil(rows, (M1 * M2));
        const auto row_lens  = make_tuple(M0, M1, M2);
#else
        constexpr index_t LdsPackPerRow = kLdsRowBytes / DWORDx4;
        constexpr index_t M2            = LdsPackPerRow / K1;
        constexpr index_t M1            = get_warp_size() / (K1 * M2);
        const index_t M0                = integer_divide_ceil(rows, (M1 * M2));
        const auto row_lens             = make_tuple(M0, number<M1>{}, number<M2>{});

        // TODO: static_assert for tests
        static_assert(K2 == 8);
        static_assert(K1 == 4);
        static_assert(M2 == 4);
        static_assert(M1 == 4);
#endif
        const auto d0 = make_naive_tensor_descriptor_packed(container_concat(row_lens, col_lens));
        const auto desc_0 = decltype(d0)(
            d0.get_transforms(), tensor_view_tmp.get_tensor_descriptor().get_element_space_size());
        const auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(M1),
                       make_merge_transform(make_tuple(number<M2>{}, number<K1>{})),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(
                sequence<0>{}, sequence<1>{}, sequence<2, 4>{}, sequence<3>{}, sequence<5>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<3>{}, sequence<2>{}, sequence<4>{}));
        constexpr index_t M2K1 = M2 * K1;
        const auto desc_2      = transform_tensor_descriptor(
            desc_1,
            make_tuple(make_pass_through_transform(M0),
                       make_xor_transform(make_tuple(M1, M2K1)),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}));
        const auto desc_3 = transform_tensor_descriptor(
            desc_2,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(M1),
                       make_unmerge_transform(make_tuple(M2, K1)),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(K2)),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<3>{}, sequence<2>{}, sequence<4>{}),
            make_tuple(
                sequence<0>{}, sequence<1>{}, sequence<2, 4>{}, sequence<3>{}, sequence<5>{}));
        const auto desc =
            transform_tensor_descriptor(desc_3,
                                        make_tuple(make_merge_transform_v3_division_mod(row_lens),
                                                   make_merge_transform_v3_division_mod(col_lens)),
                                        make_tuple(sequence<0, 1, 2>{}, sequence<3, 4, 5>{}),
                                        make_tuple(sequence<0>{}, sequence<1>{}));

        auto&& byte_ptr         = &(tensor_view_tmp.get_buffer_view()(0));
        auto&& byte_tensor_view = make_tensor_view<address_space_enum::global>(byte_ptr, desc);

        auto&& origin_tmp = window_tmp.get_window_origin();

        // Create tile distribution inline (reuse K2, K1, K0 from above)
        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
#if OLD
        constexpr index_t K1_dstr   = kElementsPerLoad;
        constexpr index_t K0_dstr   = KPerBlock / kElementsPerLoad;
#else
        constexpr index_t K1_dstr   = K2;
        constexpr index_t K0_dstr   = KPerBlock / K2;
#endif
        constexpr index_t M2_dstr   = WaveSize / K0_dstr;
        constexpr index_t M1_dstr   = BlockSize / WaveSize;
        constexpr index_t M0_dstr   = MPerBlock / (M2_dstr * M1_dstr);

        const auto tile_dstr = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<M0_dstr, M1_dstr, M2_dstr>, sequence<K0_dstr, K1_dstr>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});

        return make_tile_window(byte_tensor_view,
                                make_tuple(number<MPerBlock>{}, number<KPerBlock / APackedSize>{}),
                                {origin_tmp[0], origin_tmp[1] / APackedSize},
                                tile_dstr);
    }

    template <typename Problem, typename WindowTmp>
    CK_TILE_DEVICE static auto MakeBAsyncLoadBytesDramWindow(const WindowTmp& window_tmp)
    {
        using BDataType               = remove_cvref_t<typename Problem::BDataType>;
        constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;
        constexpr index_t NPerBlock   = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;

        constexpr auto ndims = std::decay_t<decltype(window_tmp)>::get_num_of_dimension();
        static_assert(ndims == 2, "only support 2D tensor");
        auto&& tensor_view_tmp  = window_tmp.get_bottom_tensor_view();
        const auto [rows, cols] = tensor_view_tmp.get_tensor_descriptor().get_lengths();

#if OLD
        constexpr index_t kElementsPerLoad = kMaxVecWidth / (sizeof(BDataType) * BPackedSize);
        constexpr index_t kLoadsPerBlockK  = KPerBlock / kElementsPerLoad;
        constexpr index_t K2               = kElementsPerLoad;
        constexpr index_t K1               = kLoadsPerBlockK;
        const index_t K0                   = integer_divide_ceil(cols, KPerBlock);
        const auto col_lens                = make_tuple(K0, number<K1>{}, number<K2>{});
#else
        constexpr index_t K2 = DWORDx4 / (sizeof(BDataType) * BPackedSize);
        constexpr index_t K1 = KPerBlock / K2;
        const index_t K0     = cols / (K1 * K2);
        const auto col_lens  = make_tuple(K0, number<K1>{}, number<K2>{});
#endif

#if OLD
        constexpr index_t N2 = integer_divide_ceil(kVecLoadsPerLdsRow, kLoadsPerBlockK);
        // const index_t N1     = 16;
        const index_t N1     = ck_tile::min(kVecLoadsPerLdsRow, integer_divide_ceil(rows, N2));
        const index_t N0     = integer_divide_ceil(rows, (N1 * N2));
        const auto row_lens  = make_tuple(N0, N1, N2);
#else 
        constexpr index_t LdsPackPerRow = kLdsRowBytes / DWORDx4;
        constexpr index_t N2            = LdsPackPerRow / K1;
        constexpr index_t N1            = get_warp_size() / (K1 * N2);
        const index_t N0                = integer_divide_ceil(rows, (N1 * N2));
        const auto row_lens             = make_tuple(N0, number<N1>{}, number<N2>{});
#endif

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("k, n: %d, %d, %d, %d, %d, %d\n", K2, K1, K0, N2, N1, N0);
        }
        const auto d0 = make_naive_tensor_descriptor_packed(container_concat(row_lens, col_lens));
        const auto desc_0 = decltype(d0)(
            d0.get_transforms(), tensor_view_tmp.get_tensor_descriptor().get_element_space_size());
        const auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(N0),
                       make_pass_through_transform(N1),
                       make_merge_transform(make_tuple(number<N2>{}, number<K1>{})),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(
                sequence<0>{}, sequence<1>{}, sequence<2, 4>{}, sequence<3>{}, sequence<5>{}),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<3>{}, sequence<2>{}, sequence<4>{}));
        constexpr index_t N2K1 = N2 * K1;
        const auto desc_2      = transform_tensor_descriptor(
            desc_1,
            make_tuple(make_pass_through_transform(N0),
                       make_xor_transform(make_tuple(N1, N2K1)),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}));
        const auto desc_3 = transform_tensor_descriptor(
            desc_2,
            make_tuple(make_pass_through_transform(N0),
                       make_pass_through_transform(N1),
                       make_unmerge_transform(make_tuple(N2, K1)),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(K2)),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<3>{}, sequence<2>{}, sequence<4>{}),
            make_tuple(
                sequence<0>{}, sequence<1>{}, sequence<2, 4>{}, sequence<3>{}, sequence<5>{}));
        const auto desc =
            transform_tensor_descriptor(desc_3,
                                        make_tuple(make_merge_transform_v3_division_mod(row_lens),
                                                   make_merge_transform_v3_division_mod(col_lens)),
                                        make_tuple(sequence<0, 1, 2>{}, sequence<3, 4, 5>{}),
                                        make_tuple(sequence<0>{}, sequence<1>{}));

        auto&& byte_ptr         = &(tensor_view_tmp.get_buffer_view()(0));
        auto&& byte_tensor_view = make_tensor_view<address_space_enum::global>(byte_ptr, desc);

        auto&& origin_tmp = window_tmp.get_window_origin();

        // Create tile distribution inline (reuse K2, K1, K0 from above)
        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
#if OLD
        constexpr index_t K1_dstr   = kElementsPerLoad;
        constexpr index_t K0_dstr   = KPerBlock / kElementsPerLoad;
#else
        constexpr index_t K1_dstr   = K2;
        constexpr index_t K0_dstr   = KPerBlock / K2;
#endif
        constexpr index_t N2_dstr   = WaveSize / K0_dstr;
        constexpr index_t N1_dstr   = BlockSize / WaveSize;
        constexpr index_t N0_dstr   = NPerBlock / (N2_dstr * N1_dstr);

#if !OLD
        // NOTE: We assume a wavefront can load at least one row of a block in the k dimension.
        // The expression `N2_dstr = WaveSize / K0_dstr` indicates that a wave can load at least one
        // row of a block. Therefore, KPerBlock must be smaller than 1024 (for f8) or 512 (for f16).
        // If a larger KPerBlock is required, this logic will need to be refactored.
        static_assert(KPerBlock <= kElementsPerLoad * WaveSize);
#endif
        const auto tile_dstr = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<N0_dstr, N1_dstr, N2_dstr>, sequence<K0_dstr, K1_dstr>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});

        return make_tile_window(byte_tensor_view,
                                make_tuple(number<NPerBlock>{}, number<KPerBlock / BPackedSize>{}),
                                {origin_tmp[0], origin_tmp[1] / BPackedSize},
                                tile_dstr);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        constexpr index_t vector_size =
            DS_READ_TR_SIZE() / sizeof(typename Problem::ComputeDataType);
        constexpr index_t thread_elements = WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size();
        constexpr auto wg_attr_num_access =
            !(is_a_load_tr<Problem> || is_b_load_tr<Problem>) ? WGAttrNumAccessEnum::Single
            : vector_size == thread_elements                  ? WGAttrNumAccessEnum::Single
            : vector_size * 2 == thread_elements              ? WGAttrNumAccessEnum::Double
            : vector_size * 4 == thread_elements              ? WGAttrNumAccessEnum::Quad
                                                              : WGAttrNumAccessEnum::Invalid;

        using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
                                            typename Problem::BDataType,
                                            typename Problem::CDataType, // AccDataType
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            wg_attr_num_access>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};
} // namespace ck_tile
