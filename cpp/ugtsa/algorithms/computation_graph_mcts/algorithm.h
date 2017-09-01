#pragma once

#include "ugtsa/algorithms/generalized_mcts/algorithm.h"
#include "ugtsa/computation_graphs/computation_graph/computation_graph.h"

namespace algorithms {
namespace computation_graph_mcts {

class Algorithm : public algorithms::generalized_mcts::Algorithm {
    computation_graphs::computation_graph::ComputationGraph *computation_graph;
    int empty_statistic_;
    int move_rate_;
    int game_state_as_update_;
    int updated_statistic_;
    int updated_update_;

public:
    Algorithm(games::game::GameState *game_state, int worker_count, int grow_factor, std::vector<int> removed_root_moves,
              computation_graphs::computation_graph::ComputationGraph *computation_graph, int empty_statistic, int move_rate, int game_state_as_update, int updated_statistic, int updated_update);

    Eigen::VectorXf value(int rate);

private:
    int empty_statistic(games::game::GameState *game_state);
    int move_rate(int parent_statistic, int child_statistic);
    int game_state_as_update(games::game::GameState *game_state);
    int updated_statistic(int statistic, std::vector<int> updates);
    int updated_update(int update, int statistic);
    void run_batch();
};

}
}