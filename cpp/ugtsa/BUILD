cc_library(
    name = "libugtsa",
    srcs = [
        "algorithms/algorithm/algorithm.cpp",
        "algorithms/computation_graph_mcts/algorithm.cpp",
        "algorithms/generalized_mcts/algorithm.cpp",
        "algorithms/ucb_mcts/algorithm.cpp",
        "computation_graphs/basic_computation_graph/computation_graph.cpp",
        "computation_graphs/computation_graph/computation_graph.cpp",
        "computation_graphs/synchronized_computation_graph/computation_graph.cpp",
        "games/game/game_state.cpp",
        "games/omringa/game_state.cpp",
        "common.cpp",
    ],
    hdrs = [
        "algorithms/algorithm/algorithm.h",
        "algorithms/computation_graph_mcts/algorithm.h",
        "algorithms/generalized_mcts/algorithm.h",
        "algorithms/ucb_mcts/algorithm.h",
        "computation_graphs/basic_computation_graph/computation_graph.h",
        "computation_graphs/computation_graph/computation_graph.h",
        "computation_graphs/synchronized_computation_graph/computation_graph.h",
        "games/game/game_state.h",
        "games/omringa/game_state.h",
        "common.h",
    ],
    deps = [
        "//tensorflow/core:tensorflow",
        "//third_party/eigen3",
        "@boost//:filesystem",
    ]
)

cc_binary(
    name = "ugtsa",
    srcs = ["main.cpp"],
    deps = [
        ":libugtsa",
        "//tensorflow/core:tensorflow",
        "//third_party/eigen3",
    ]
)

cc_binary(
    name = "solution_001",
    srcs = ["solution_001.cpp"],
    deps = [
        ":libugtsa",
        "//tensorflow/core:tensorflow",
        "//third_party/eigen3",
    ]
)

cc_binary(
    name = "speed_test_1",
    srcs = ["speed_test_1.cpp"],
    deps = [
        ":libugtsa",
        "//tensorflow/core:tensorflow",
        "//third_party/eigen3",
    ]
)

cc_binary(
    name = "speed_test_2",
    srcs = ["speed_test_2.cpp"],
    deps = [
        ":libugtsa",
        "//tensorflow/core:tensorflow",
        "//third_party/eigen3",
    ]
)