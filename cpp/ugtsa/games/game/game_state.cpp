#include "ugtsa/algorithms/ucb_mcts/algorithm.h"
#include "ugtsa/games/game/game_state.h"

#include <cstdlib>
#include <iostream>

namespace games {
namespace game {

Eigen::VectorXf GameState::random_playout_payoff() {
    int counter = 0;

    while (!is_final()) {
        apply_move(std::uniform_int_distribution<int>(0, move_count() - 1)(generator));
        counter++;
    }

    auto payoff = this->payoff();

    for (int i = 0; i < counter; i++) {
        undo_move();
    }

    return payoff;
}

void GameState::move_to_random_state() {
    int counter = 0;

    while (!is_final()) {
        if (player == -1) {
            apply_move(std::uniform_int_distribution<int>(0, move_count() - 1)(generator));
        } else {
            auto game_state = std::unique_ptr<GameState>(copy());
            auto algorithm = algorithms::ucb_mcts::Algorithm(game_state.get(), 10, 5, {}, std::sqrt(2.));
            for (int i = 0; i < 3000; i++) {
                algorithm.improve();
            }
            apply_move(algorithm.best_move());
        }
        counter += 1;
    }

    int undo_times = std::uniform_int_distribution<int>(0, counter)(generator);

    for (int i = 0; i < undo_times; i++) {
        undo_move();
    }
}

std::ostream& operator<<(std::ostream& stream, const GameState& game_state) {
    game_state.serialize(stream);
    return stream;
}

}
}
