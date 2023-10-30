#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_impl_ht.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using F16 = ck::half_t;
using F32 = float;

using ADataType = F32;
using BDataType = F32;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using UnaryOp     = ck::tensor_operation::element_wise::PassThrough;
// ck::index_t scalar_mult = 1;

using DeviceElementwisePermuteInstance =
    ck::tensor_operation::device::DeviceElementwiseImpl<ck::Tuple<ADataType>, // InDataTypeTuple
                                                        ck::Tuple<BDataType>, // OutDataTypeTuple
                                                        PassThrough,          // ElementwiseOp
                                                        UnaryOp,              // UnaryOp
                                                        4,                    // NumDim
                                                        8,                    // MPerThread
                                                        1,                    // ScalarMult (alpha)
                                                        ck::Sequence<8>,  // InScalarPerVectorSeq
                                                        ck::Sequence<1>>; // OutScalarPerVectorSeq

template <typename HostTensorA, typename HostTensorB, typename FunctorA, typename FunctorB>
void host_elementwise4D(HostTensorB& B_cnhw,
                        const HostTensorA& A_whcn,
                        FunctorA functor_a,
                        FunctorB functor_b)
{
    for(std::size_t w = 0; w < A_whcn.mDesc.GetLengths()[0]; ++w)
        for(std::size_t h = 0; h < A_whcn.mDesc.GetLengths()[1]; ++h)
            for(std::size_t c = 0; c < A_whcn.mDesc.GetLengths()[2]; ++c)
                for(std::size_t n = 0; n < A_whcn.mDesc.GetLengths()[3]; ++n)
                {
                    ADataType tmp_val;
                    auto a_val = A_whcn(w, h, c, n);
                    functor_b(tmp_val, a_val);
                    functor_a(B_cnhw(c, n, h, w), tmp_val);
                }
}

int main()
{
    bool do_verification = true;
    bool time_kernel     = true;

    std::vector<std::size_t> whcn = {3, 2, 4, 5}; // whcn : n => w, c => h, h => c, w => n
    std::vector<std::size_t> cnhw = {4, 5, 2, 3};
    Tensor<ADataType> a(whcn);
    Tensor<BDataType> b(cnhw);

    // a.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
    std::iota(a.mData.begin(), a.mData.end(), (ADataType)0.0);

    printf("a data: \n");
    for (auto v : a.mData) { printf("%f, ", v);}
    printf("\n");

    DeviceMem a_device_buf(sizeof(ADataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a.mData.data());

    std::array<const void*, 1> input = {a_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {b_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 4> ab_lengths;
    std::array<ck::index_t, 4> a_strides = {
	    1,
	    static_cast<int>(whcn[0]),
	    static_cast<int>(whcn[0] * whcn[1]),
	    static_cast<int>(whcn[0] * whcn[1] * whcn[2]),
    };
    std::array<ck::index_t, 4> b_strides = {
	    static_cast<int>(cnhw[0] * cnhw[1] * cnhw[2]),
	    static_cast<int>(cnhw[0] * cnhw[1]),
	    1,
	    static_cast<int>(cnhw[0]),
    };

    ck::ranges::copy(whcn, ab_lengths.begin());

    auto broadcastPermute = DeviceElementwisePermuteInstance{};
    auto argument         = broadcastPermute.MakeArgumentPointer(
        ab_lengths, {a_strides}, {b_strides}, input, output, PassThrough{}, UnaryOp{});

    /*
    if(!broadcastPermute.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };
    */

    std::cout << "A (whcn): " << a.mDesc << std::endl;
    std::cout << "B (cnhw): " << b.mDesc << std::endl;

    auto broadcastPermute_invoker_ptr = broadcastPermute.MakeInvokerPointer();
    float ave_time =
        broadcastPermute_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});

    std::size_t flop = std::size_t(2) * whcn[0] * whcn[1] * whcn[2] * whcn[3];

    std::size_t num_btype = sizeof(ADataType) * (whcn[0] * whcn[1] * whcn[2] * whcn[3]) +
                            sizeof(BDataType) * (whcn[0] * whcn[1] * whcn[2] * whcn[3]);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    bool pass = true;

    if(do_verification)
    {
        b_device_buf.FromDevice(b.mData.data());
    printf("b data: \n");
    for (auto v : b.mData) { printf("%f, ", v);}
    printf("\n");

        Tensor<BDataType> host_b(cnhw);
        host_elementwise4D(host_b, a, PassThrough{}, UnaryOp{});

        pass &=
            ck::utils::check_err(b.mData, host_b.mData, "Error: Incorrect results b", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
