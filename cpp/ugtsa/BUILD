cc_library(
    name = "foo",
    srcs = [
        "algorithms/algorithm/algorithm.cpp",
        "algorithms/computation_graph_mcts/algorithm.cpp",
        "algorithms/generalized_mcts/algorithm.cpp",
        "algorithms/ucb_mcts/algorithm.cpp",
        "computation_graphs/basic_computation_graph/computation_graph.cpp",
        "computation_graphs/computation_graph/computation_graph.cpp",
        "games/game/game_state.cpp",
        "games/omringa/game_state.cpp",
    ],
    hdrs = [
        "algorithms/algorithm/algorithm.h",
        "algorithms/computation_graph_mcts/algorithm.h",
        "algorithms/generalized_mcts/algorithm.h",
        "algorithms/ucb_mcts/algorithm.h",
        "computation_graphs/basic_computation_graph/computation_graph.h",
        "computation_graphs/computation_graph/computation_graph.h",
        "games/game/game_state.h",
        "games/omringa/game_state.h",
    ],
    deps = [
        "//tensorflow/core:tensorflow",
        "//third_party/eigen3"
    ]
)

cc_binary(
    name = "ugtsa",
    srcs = ["main.cpp"],
    deps = [
        ":foo",
        "//tensorflow/core:tensorflow",
        "//third_party/eigen3",
    ]
)