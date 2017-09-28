#include "ugtsa/algorithms/computation_graph_mcts/algorithm.h"

namespace algorithms {
namespace computation_graph_mcts {

Algorithm::Algorithm(games::game::GameState *game_state, int worker_count, int grow_factor, std::vector<int> removed_root_moves,
                     computation_graphs::computation_graph::ComputationGraph *computation_graph, int empty_statistic, int move_rate, int game_state_as_update, int updated_statistic, int updated_update)
    : algorithms::generalized_mcts::Algorithm(game_state, worker_count, grow_factor, removed_root_moves), computation_graph(computation_graph),
      empty_statistic_(empty_statistic), move_rate_(move_rate), game_state_as_update_(game_state_as_update), updated_statistic_(updated_statistic), updated_update_(updated_update) {}

int Algorithm::empty_statistic(games::game::GameState *game_state) {
    return computation_graph->transformation_run(empty_statistic_, {
        {computation_graph->matrix(game_state->matrix())},
        {computation_graph->matrix(game_state->statistic())}
    });
}

int Algorithm::move_rate(int parent_statistic, int child_statistic) {
    return computation_graph->transformation_run(move_rate_, {{parent_statistic}, {child_statistic}});
}

int Algorithm::game_state_as_update(games::game::GameState *game_state) {
    return computation_graph->transformation_run(game_state_as_update_, {
        {computation_graph->matrix(game_state->update_statistic())}
    });
}

int Algorithm::updated_statistic(int statistic, std::vector<int> updates) {
    Eigen::VectorXi updates_size(1);
    updates_size << updates.size();
    return computation_graph->transformation_run(updated_statistic_, {
        {statistic},
        {computation_graph->matrix(updates_size)},
        {updates}
    });
}

int Algorithm::updated_update(int update, int statistic) {
    return computation_graph->transformation_run(updated_update_, {
        {update},
        {statistic}
    });
}

void Algorithm::run_batch() {
    computation_graph->run_batch();
}

Eigen::VectorXf Algorithm::value(int rate) const {
    return computation_graph->value(rate);
}

}
}