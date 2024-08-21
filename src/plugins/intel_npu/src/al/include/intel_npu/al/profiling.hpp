// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/runtime/profiling_info.hpp"

namespace intel_npu::profiling {

using LayerStatistics = std::vector<ov::ProfilingInfo>;

template <typename LayerInfo>
LayerStatistics convertLayersToIeProfilingInfo(const std::vector<LayerInfo>& layerInfo) {
    LayerStatistics perfCounts;
    printf(" Debug - convertLayersToIeProfilingInfo() start 1 \n");
    perfCounts.reserve(layerInfo.size());
    for (const auto& layer : layerInfo) {
        ov::ProfilingInfo& info = perfCounts.emplace_back();
        info.status = ov::ProfilingInfo::Status::EXECUTED;
        const auto real_time_ns = std::chrono::nanoseconds(layer.duration_ns);
        info.real_time = std::chrono::duration_cast<std::chrono::microseconds>(real_time_ns);
        const auto cpu_time_ns = std::chrono::nanoseconds(layer.dma_ns + layer.sw_ns + layer.dpu_ns);
        info.cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_time_ns);
        info.node_name = layer.name;
        printf(" Debug - convertLayersToIeProfilingInfo() running 1 \n");
        if (layer.sw_ns > 0) {
            info.exec_type = "Shave";
        } else if (layer.dpu_ns > 0) {
            info.exec_type = "DPU";
        } else {
            info.exec_type = "DMA";
        }
        info.node_type = layer.layer_type;
    }
    printf(" Debug - convertLayersToIeProfilingInfo() start 2 \n");
    return perfCounts;
}

}  // namespace intel_npu::profiling
