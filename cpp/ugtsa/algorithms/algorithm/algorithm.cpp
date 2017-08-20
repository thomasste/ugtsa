#include "algorithms/algorithm/algorithm.h"

namespace algorithms {
namespace algorithm {

Algorithm::Algorithm(games::game::GameState *game_state) : game_state(game_state) {}

int Algorithm::best_move() {
    assert(!game_state->is_final());
    assert(game_state->player != -1);

    int best_move = -1;
    float best_rate = -std::numeric_limits<float>::infinity();

    auto move_rates = this->move_rates();

    for (int i = 0; i < move_rates.size(); i++) {
        auto move_rate = value(move_rates[i]);
        if (best_rate < (*move_rate)(game_state->player)) {
            best_rate = (*move_rate)(game_state->player);
            best_move = i;
        }
    }

    return best_move;
}

std::ostream& operator<<(std::ostream& stream, const Algorithm& algorithm) {
    algorithm.serialize(stream);
    return stream;
}

}
}
