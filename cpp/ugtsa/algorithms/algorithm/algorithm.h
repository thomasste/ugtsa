#pragma once

#include "ugtsa/games/game/game_state.h"

#include <limits>
#include <vector>

namespace algorithms {
namespace algorithm {

class Algorithm {
public:
    games::game::GameState *game_state;

    Algorithm(games::game::GameState *game_state);

    virtual void improve() = 0;
    virtual std::vector<int> move_rates() = 0;
    virtual Eigen::VectorXf *value(int rate) = 0;

    int best_move();

    virtual void serialize(std::ostream& stream) const = 0;

    friend std::ostream& operator<<(std::ostream& stream, const Algorithm& algorithm);
};

std::ostream& operator<<(std::ostream& stream, const Algorithm& algorithm);

}
}
