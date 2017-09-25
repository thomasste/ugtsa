#include "tensorflow/core/public/session.h"

#include "ugtsa/algorithms/ucb_mcts/algorithm.h"
#include "ugtsa/computation_graphs/basic_computation_graph/computation_graph.h"
#include "ugtsa/games/omringa/game_state.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    srand(std::atoi(argv[1]));

    auto gs = games::omringa::GameState();
    gs.move_to_random_state();

    std::cout << gs << std::endl;
    std::cout << gs.random_playout_payoff() << std::endl;
    std::cout << gs.move_count() << std::endl;

    std::default_random_engine generator;

    while (!gs.is_final()) {
        std::cout << gs << std::endl;
        if (gs.player == -1) {
            gs.apply_move(std::uniform_int_distribution<int>(0, gs.move_count() - 1)(generator));
        } else {
            auto a = algorithms::ucb_mcts::Algorithm(&gs, 10, 5, std::vector<int>(), std::sqrt(2.));
            for (int i = 0; i < 500000; i++) {
                a.improve();
            }
            std::cout << a << std::endl;
            gs.apply_move(a.best_move());
        }
    }

    std::cout << gs.payoff() << std::endl;

    return 0;
}
