#pragma once

#include "ugtsa/computation_graphs/computation_graph/computation_graph.h"
#include <vector>
#include <string>

namespace common {
namespace model_helpers {

void load_model(tensorflow::Session* session, std::string graph_name);
void store_model(tensorflow::Session* session, std::string graph_name);

int worker_count(tensorflow::Session* session);
int global_step(tensorflow::Session* session);

std::vector<int> create_transformations(tensorflow::Session* session, computation_graphs::computation_graph::ComputationGraph *computation_graph);

void zero_model_gradient_accumulators(tensorflow::Session* session);
std::pair<float, std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>>> cost_function(
    tensorflow::Session* session,
    std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> move_rates,
    std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> ucb_move_rates,
    std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> ugtsa_move_rates);
void apply_gradients(tensorflow::Session* session);

}
}