#include "ugtsa/algorithms/computation_graph_mcts.h"

namespace algorithms {
namespace computation_graph_mcts {

Algorithm::Algorithm(games::game::GameState *game_state, int worker_count, int grow_factor, std::vector<int> removed_root_moves,
                     ComputationGraph *computation_graph, int empty_statistic, int move_rate, int game_state_as_update, int updated_statistic, int updated_update)
    : algorithms::generalized_mcts::Algorithm(game_state, worker_count, grow_factor, removed_root_moves), computation_graph(computation_graph),
      empty_statistic(empty_statistic), move_rate(move_rate), game_state_as_update(game_state_as_update), updated_update(updated_update) {}

int Algorithm::empty_statistic(games::game::GameState *game_state) {
    return computation_graph->transformation_run(empty_statistic, {
        {computation_graph->matrix(game_state->as_matrix())},
        {computation_graph->matrix(game_state->as_statistics())}
    });
}

int Algorithm::move_rate(int parent_statistic, int child_statistic) {
    return computation_graph->transformation_run(move_rate, {{parent_statistic}, {child_statistic}});
}

int Algorithm::game_state_as_update(games::game::GameState *game_state) {
    return computation_graph->transformation_run(game_state_as_update, {
        {computation_graph->matrix(game_state->as_update_statistic())}
    });
}

int Algorithm::updated_statistic(int statistic, std::vector<int> updates) {
    return computation_graph->transformation_run(updated_statistic, {
        {statistic},
        {computation_graph->matrix(Eigen::VectorXi::Constant(updates.size()))},
        {updates}
    });
}

int Algorithm::updated_update(int update, int statistic) {
    return computation_graph->transformation_run(updated_update, {
        {update},
        {statistic}
    });
}

void Algorithm::run_batch() {
    computation_graph->run_batch();
}

Eigen::VectorXf value(int rate) {
    return computation_graph->value(rate);
}

}
}