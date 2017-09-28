#pragma once

#include "ugtsa/algorithms/generalized_mcts/algorithm.h"

namespace algorithms {
namespace ucb_mcts {

class Algorithm : public algorithms::generalized_mcts::Algorithm {
    std::vector<int> ns;
    std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> ws;
    std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> move_rates_;
    std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> updates;

    float exploration_factor;

public:
    Algorithm(games::game::GameState *game_state, int worker_count, int grow_factor, std::vector<int> removed_root_moves, float exploration_factor);

    Eigen::VectorXf value(int rate) const;

    friend std::ostream& operator<<(std::ostream& stream, const Algorithm& algorithm);

private:
    int empty_statistic(games::game::GameState *game_state);
    int move_rate(int parent_statistic, int child_statistic);
    int game_state_as_update(games::game::GameState *game_state);
    int updated_statistic(int statistic, std::vector<int> updates);
    int updated_update(int update, int statistic);
    void run_batch();
};

std::ostream& operator<<(std::ostream& stream, const Algorithm& algorithm);

}
}
