// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_multi_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/batched_gemm.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm_multi_d.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"

namespace ck {
namespace profiler {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementOp,
          typename BElementOp,
          typename CElementOp,
          typename DeviceOp>
bool profile_gemm_universal_batched_impl(int do_verification,
                                         int init_method,
                                         bool do_log,
                                         bool time_kernel,
                                         int M,
                                         int N,
                                         int K,
                                         int BatchStrideA,
                                         int BatchStrideB,
                                         int BatchStrideC,
                                         int StrideA,
                                         int StrideB,
                                         int StrideC,
                                         int BatchCount,
                                         int KBatch,
                                         int n_warmup,
                                         int n_iter,
                                         uint64_t rotating = 0)
{
    bool pass = true;

    auto f_host_tensor_descriptor = [](std::size_t batch_count,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       std::size_t batch_stride,
                                       auto layout) {
        using namespace ck::literals;

        if(is_same<decltype(layout), tensor_layout::gemm::RowMajor>::value)
        {
            return HostTensorDescriptor({batch_count, row, col}, {batch_stride, stride, 1_uz});
        }
        else
        {
            return HostTensorDescriptor({batch_count, row, col}, {batch_stride, 1_uz, stride});
        }
    };

    Tensor<ADataType> a_g_m_k(
        f_host_tensor_descriptor(BatchCount, M, K, StrideA, BatchStrideA, ALayout{}));
    Tensor<BDataType> b_g_k_n(
        f_host_tensor_descriptor(BatchCount, K, N, StrideB, BatchStrideB, BLayout{}));
    Tensor<CDataType> c_g_m_n_host_result(
        f_host_tensor_descriptor(BatchCount, M, N, StrideC, BatchStrideC, CLayout{}));
    Tensor<CDataType> c_g_m_n_device_result(
        f_host_tensor_descriptor(BatchCount, M, N, StrideC, BatchStrideC, CLayout{}));

    int total_gemm_needed =
        a_g_m_k.GetElementSpaceSizeInBytes() + b_g_k_n.GetElementSpaceSizeInBytes();
    int rotating_count = std::max(
        1,
        std::min(n_iter,
                 static_cast<int>(std::ceil(static_cast<double>(rotating) / total_gemm_needed))));

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b_g_k_n: " << b_g_k_n.mDesc << std::endl;
    std::cout << "c_g_m_n: " << c_g_m_n_host_result.mDesc << std::endl;
    std::cout << "rotating count: " << rotating_count << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    default:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_g_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
    }

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};

    if(do_verification)
    {
        using ReferenceBatchedGemmInstance =
            ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                             BDataType,
                                                             CDataType,
                                                             float,
                                                             AElementOp,
                                                             BElementOp,
                                                             CElementOp>;

        auto ref_batched_gemm = ReferenceBatchedGemmInstance{};
        auto ref_invoker      = ref_batched_gemm.MakeInvoker();

        auto ref_argument = ref_batched_gemm.MakeArgument(
            a_g_m_k, b_g_k_n, c_g_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_g_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_g_m_n_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_g_m_k.mData.data());
    b_device_buf.ToDevice(b_g_k_n.mData.data());
    c_device_buf.ToDevice(c_g_m_n_device_result.mData.data());

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    float best_kbatch     = 0;

    // profile device op instances
    for(auto& op_ptr : op_ptrs)
    {
        std::vector<int> kbatch_list = {1, 2, 4, 8, 16, 19, 32, 38};

        if(KBatch > 0)
        {
            kbatch_list = {KBatch};
        }

        for(std::size_t i = 0; i < kbatch_list.size(); i++)
        {
            auto kbatch_curr = kbatch_list[i];

            auto argument_ptr =
                op_ptr->MakeArgumentPointer(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                            static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                            {},
                                            static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                            M,
                                            N,
                                            K,
                                            BatchCount,
                                            StrideA,
                                            StrideB,
                                            {},
                                            StrideC,
                                            BatchStrideA,
                                            BatchStrideB,
                                            {},
                                            BatchStrideC,
                                            ck::tensor_operation::element_wise::PassThrough{},
                                            ck::tensor_operation::element_wise::PassThrough{},
                                            ck::tensor_operation::element_wise::PassThrough{},
                                            kbatch_curr);

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                std::string op_name = op_ptr->GetTypeString();

                float ave_time = invoker_ptr->Run(
                    argument_ptr.get(),
                    StreamConfig{nullptr, time_kernel, 0, n_warmup, n_iter, true, rotating_count});

                std::size_t flop = std::size_t(2) * BatchCount * M * N * K;

                std::size_t num_btype = (sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                         sizeof(CDataType) * M * N) *
                                        BatchCount;

                float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

                float gb_per_sec = num_btype / 1.E6 / ave_time;

                std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                          << " GB/s, " << op_name << ", KBatch " << kbatch_curr << std::endl;

                if(tflops > best_tflops)
                {
                    best_op_name    = op_name;
                    best_tflops     = tflops;
                    best_ave_time   = ave_time;
                    best_gb_per_sec = gb_per_sec;
                    best_kbatch     = kbatch_curr;
                }

                if(do_verification)
                {
                    c_device_buf.FromDevice(c_g_m_n_device_result.mData.data());

                    pass = pass & ck::utils::check_err(c_g_m_n_device_result, c_g_m_n_host_result);

                    if(do_log)
                    {
                        LogRangeAsType<float>(std::cout << "a : ", a_g_m_k.mData, ",") << std::endl;
                        LogRangeAsType<float>(std::cout << "b: ", b_g_k_n.mData, ",") << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "c_host: ", c_g_m_n_host_result.mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "c_device: ", c_g_m_n_device_result.mData, ",")
                            << std::endl;
                    }
                }
            }
            else
            {
                std::cout << op_ptr->GetTypeString() << " does not support this problem"
                          << std::endl;
            }
        }
    }

    if constexpr(is_same<CDataType, float>::value)
    {
        std::cout << "Best Perf for datatype = f32";
    }
    else if constexpr(is_same<CDataType, half_t>::value)
    {
        std::cout << "Best Perf for datatype = f16";
    }
    else if constexpr(is_same<CDataType, bhalf_t>::value)
    {
        std::cout << "Best Perf for datatype = bf16";
    }
    else if constexpr(is_same<CDataType, int8_t>::value)
    {
        std::cout << "Best Perf for datatype = int8";
    }

    if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value)
    {
        std::cout << " ALayout =  RowMajor";
    }
    else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value)
    {
        std::cout << " ALayout =  ColumnMajor";
    }

    if constexpr(is_same<BLayout, tensor_layout::gemm::RowMajor>::value)
    {
        std::cout << " BLayout =  RowMajor";
    }
    else if constexpr(is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value)
    {
        std::cout << " BLayout =  ColumnMajor";
    }

    std::cout << " B = " << BatchCount << " M = " << M << " N = " << N << " K = " << K
              << " StrideA = " << StrideA << " StrideB = " << StrideB << " StrideC = " << StrideC
              << " KBatch = " << best_kbatch << ": " << best_ave_time << " ms, " << best_tflops
              << " TFlops, " << best_gb_per_sec << " GB/s, " << best_op_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
