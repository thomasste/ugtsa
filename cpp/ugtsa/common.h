#pragma once

#include "ugtsa/computation_graphs/computation_graph/computation_graph.h"
#include <vector>
#include <string>

namespace common {

void load_model(tensorflow::Session* session, std::string graph_name);
void store_model(tensorflow::Session* session, std::string graph_name);

std::vector<int> create_transformations(
    computation_graphs::computation_graph::ComputationGraph *computation_graph,
    int player_count,
    int worker_count,
    int statistic_size,
    int update_size,
    std::vector<int> game_state_board_shape,
    int game_state_statistic_size,
    int update_statistic_size,
    int seed_size);

void zero_model_gradient_accumulators(tensorflow::Session* session);
std::pair<float, std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>>> cost_function(
    tensorflow::Session* session,
    std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> move_rates,
    std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> ucb_move_rates,
    std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> ugtsa_move_rates);
void apply_gradients(tensorflow::Session* session);

}