#include "ugtsa/algorithms/ucb_mcts/algorithm.h"

#include <iostream>

namespace algorithms {
namespace ucb_mcts {

Algorithm::Algorithm(games::game::GameState *game_state, int worker_count, int grow_factor, float move_choice_factor, std::vector<int> removed_root_moves, float exploration_factor)
    : algorithms::generalized_mcts::Algorithm(game_state, worker_count, grow_factor, move_choice_factor, removed_root_moves), exploration_factor(exploration_factor) {}

Eigen::VectorXf Algorithm::value(int rate) const {
    return move_rates_[rate];
}

int Algorithm::empty_statistic(games::game::GameState *game_state) {
    ns.push_back(0);
    ws.push_back(Eigen::VectorXf::Zero(game_state->player_count));

    return ws.size() - 1;
}

int Algorithm::move_rate(int parent_statistic, int child_statistic) {
    int pn = ns[parent_statistic];
    int cn = ns[child_statistic];
    auto &cw = ws[child_statistic];

    move_rates_.push_back(
        cw / ((float) cn + 0.1) + exploration_factor * std::sqrt(std::log((float) pn + 0.1) / ((float) cn + 0.1)) * Eigen::VectorXf::Ones(cw.size()));

    return move_rates_.size() - 1;
}

int Algorithm::game_state_as_update(games::game::GameState *game_state) {
    updates.push_back(game_state->random_playout_payoff());

    return updates.size() - 1;
}

int Algorithm::updated_statistic(int statistic, std::vector<int> updates) {
    auto w = Eigen::MatrixXf(ws[statistic]);
    int n = ns[statistic];

    for (auto update : updates) {
        auto &payoff = this->updates[update];

        float best_score = -std::numeric_limits<float>::infinity();
        int best_player = -1;
        for (int i = 0; i < payoff.size(); i++) {
            if (best_score < payoff(i)) {
                best_score = payoff(i);
                best_player = i;
            }
        }
        w(best_player) += 1.;
    }

    ns.push_back(n + updates.size());
    ws.push_back(w);

    return ns.size() - 1;
}

int Algorithm::updated_update(int update, int statistic) {
    return update;
}

void Algorithm::run_batch() {
}

std::ostream& operator<<(std::ostream& stream, const Algorithm& algorithm) {
    algorithm.serialize(stream);
    return stream;
}

}
}
