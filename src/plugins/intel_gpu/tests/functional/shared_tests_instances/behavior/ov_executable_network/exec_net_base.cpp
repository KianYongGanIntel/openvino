// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/compiled_model_base.hpp"

using namespace ov::test::behavior;
namespace {
auto configs = []() {
    return std::vector<ov::AnyMap>{
        {},
    };
};

auto autoBatchConfigs = []() {
    return std::vector<ov::AnyMap>{
        // explicit batch size 4 to avoid fallback to no auto-batching (i.e. plain GPU)
        {{CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG), std::string(ov::test::utils::DEVICE_GPU) + "(4)"},
         // no timeout to avoid increasing the test time
         {CONFIG_KEY(AUTO_BATCH_TIMEOUT), "0 "}}};
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVCompiledModelBaseTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_GPU),
                                ::testing::ValuesIn(configs())),
                        OVCompiledModelBaseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatchBehaviorTests, OVCompiledModelBaseTest,
                         ::testing::Combine(
                                 ::testing::Values(ov::test::utils::DEVICE_BATCH),
                                 ::testing::ValuesIn(autoBatchConfigs())),
                         OVCompiledModelBaseTest::getTestCaseName);
}  // namespace
