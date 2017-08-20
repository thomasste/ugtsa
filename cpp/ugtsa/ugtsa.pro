TEMPLATE = app
CONFIG += console c++14
CONFIG += object_parallel_to_source
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    games/game/game_state.cpp \
    algorithms/algorithm/algorithm.cpp \
    algorithms/generalized_mcts/algorithm.cpp \
    algorithms/ucb_mcts/algorithm.cpp \
    games/omringa/game_state.cpp

HEADERS += \
    algorithms/algorithm/algorithm.h \
    games/game/game_state.h \
    algorithms/generalized_mcts/algorithm.h \
    algorithms/ucb_mcts/algorithm.h \
    games/omringa/game_state.h
