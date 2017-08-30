cc_library(
    name = "foo",
    srcs = [
        "algorithms/algorithm/algorithm.cpp",
        "algorithms/generalized_mcts/algorithm.cpp",
        "algorithms/ucb_mcts/algorithm.cpp",
        "games/game/game_state.cpp",
        "games/omringa/game_state.cpp",
    ],
    hdrs = [
        "algorithms/algorithm/algorithm.h",
        "algorithms/generalized_mcts/algorithm.h",
        "algorithms/ucb_mcts/algorithm.h",
        "games/game/game_state.h",
        "games/omringa/game_state.h",
    ],
)

cc_binary(
    name = "ugtsa",
    srcs = ["main.cpp"],
    deps = [
        "//tensorflow/core:tensorflow",
        ":foo",
    ]
)