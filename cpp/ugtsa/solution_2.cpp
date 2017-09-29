#include "tensorflow/core/public/session.h"

#include <thread>
#include "ugtsa/common/model_helpers.h"
#include "ugtsa/algorithms/ucb_mcts/algorithm.h"
#include "ugtsa/algorithms/computation_graph_mcts/algorithm.h"
#include "ugtsa/computation_graphs/basic_computation_graph/computation_graph.h"
#include "ugtsa/computation_graphs/synchronized_computation_graph/computation_graph.h"
#include "ugtsa/games/omringa/game_state.h"

std::string graph_name;
int grow_factor;
float move_choice_factor;
int ucb_strength;
int ugtsa_strength;
int number_of_iterations;
int store_model_each;
int thread_count;
int model_updates_per_state;

struct {
    std::unique_ptr<games::game::GameState> game_state;
    std::unique_ptr<algorithms::algorithm::Algorithm> algorithm;
    std::vector<int> move_rates;
} contexts[100];

void improve_ucb_algorithm(int index) {
    auto &context = contexts[index];
    for (int i = 0; i < ucb_strength / model_updates_per_state; i++) {
        context.algorithm->improve();
    }
    context.move_rates = context.algorithm->move_rates();
}

void improve_training_algorithm(int index) {
    auto &context = contexts[index];
    for (int i = 0; i < ugtsa_strength / model_updates_per_state; i++) {
        context.algorithm->improve();
    }
    context.move_rates = context.algorithm->move_rates();
}

void improve_oracle_algorithm(int index) {
    auto &context = contexts[index];
    for (int i = 0; i < 2 * (ugtsa_strength / model_updates_per_state); i++) {
        context.algorithm->improve();
    }
    context.move_rates = context.algorithm->move_rates();
}

int main(int argc, char **argv) {
    srand(time(NULL));

    // omringa__default__big_vertical_lstm_model_builder__10 5 80. 500 500 1 1 1 1
    // omringa__default__small_vertical_lstm_model_builder__10 5 80. 500 500 1 1 1 1
    graph_name = argv[1];
    grow_factor = std::atoi(argv[2]);
    move_choice_factor = std::atof(argv[3]);
    ucb_strength = std::atoi(argv[4]);
    ugtsa_strength = std::atoi(argv[5]);
    number_of_iterations = std::atoi(argv[6]);
    store_model_each = std::atoi(argv[7]);
    thread_count = std::atoi(argv[8]);
    model_updates_per_state = std::atoi(argv[9]);

    // create session
    tensorflow::Session* session;
    tensorflow::SessionOptions session_options = tensorflow::SessionOptions();
    TF_CHECK_OK(tensorflow::NewSession(session_options, &session));

    // load graph
    common::model_helpers::load_model(session, graph_name);
    int worker_count = common::model_helpers::worker_count(session);

    for (int i = 0; i < number_of_iterations; i++) {
        std::cout << "global step " << common::model_helpers::global_step(session) << std::endl;

        // computation graphs
        auto oracle_computation_graph_ = computation_graphs::basic_computation_graph::ComputationGraph(
            session, "training:0", false);
        auto training_computation_graph_ = computation_graphs::basic_computation_graph::ComputationGraph(
            session, "training:0", true);
        computation_graphs::synchronized_computation_graph::ComputationGraph oracle_computation_graph(&oracle_computation_graph_, thread_count);
        computation_graphs::synchronized_computation_graph::ComputationGraph training_computation_graph(&training_computation_graph_, thread_count);
        auto oracle_transformations = common::model_helpers::create_transformations(session, &oracle_computation_graph);
        auto training_transformations = common::model_helpers::create_transformations(session, &training_computation_graph);
        int backprop_offset = 0;

        // prepare threads
        for (int j = 0; j < thread_count; j++) {
            auto game_state = games::omringa::GameState(); game_state.move_to_random_state(); if (game_state.is_final()) game_state.undo_move();
            // ucb
            {
                auto &context = contexts[j * 3];
                context.game_state = std::unique_ptr<games::game::GameState>(game_state.copy());
                context.algorithm = std::unique_ptr<algorithms::algorithm::Algorithm>(
                    new algorithms::ucb_mcts::Algorithm(
                        context.game_state.get(), worker_count, grow_factor, move_choice_factor, {}, std::sqrt(2.)));
            }
            // oracle
            {
                auto &context = contexts[j * 3 + 1];
                context.game_state = std::unique_ptr<games::game::GameState>(game_state.copy());
                context.algorithm = std::unique_ptr<algorithms::algorithm::Algorithm>(
                    new algorithms::computation_graph_mcts::Algorithm(
                        context.game_state.get(), worker_count, grow_factor, move_choice_factor, {}, &oracle_computation_graph,
                        oracle_transformations[0], oracle_transformations[1], oracle_transformations[2], oracle_transformations[3], oracle_transformations[4]));
            }
            // training
            {
                auto &context = contexts[j * 3 + 2];
                context.game_state = std::unique_ptr<games::game::GameState>(game_state.copy());
                context.algorithm = std::unique_ptr<algorithms::algorithm::Algorithm>(
                    new algorithms::computation_graph_mcts::Algorithm(
                        context.game_state.get(), worker_count, grow_factor, move_choice_factor, {}, &training_computation_graph,
                        training_transformations[0], training_transformations[1], training_transformations[2], training_transformations[3], training_transformations[4]));
            }
        }

        for (int j = 0; j < model_updates_per_state; j++) {
            // improve algorithms
            std::vector<std::thread> threads;
            for (int k = 0; k < thread_count; k++) {
                threads.push_back(std::thread(improve_ucb_algorithm, 3 * k));
                threads.push_back(std::thread(improve_oracle_algorithm, 3 * k + 1));
                threads.push_back(std::thread(improve_training_algorithm, 3 * k + 2));
            }
            for (auto &thread : threads) {
                thread.join();
            }
            // collect training data
            std::vector<int> training_move_rates;
            std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> ucb_move_rate_values;
            std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> oracle_move_rate_values;
            std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> training_move_rate_values;
            for (int k = 0; k < thread_count; k++) {
                auto &ucb_context = contexts[3 * k];
                auto &oracle_context = contexts[3 * k + 1];
                auto &training_context = contexts[3 * k + 2];

                // std::cout << *ucb_context.algorithm << std::endl;
                // std::cout << *oracle_context.algorithm << std::endl;
                // std::cout << *training_context.algorithm << std::endl;

                for (auto &x : training_context.move_rates) training_move_rates.push_back(x);
                for (auto &x : ucb_context.move_rates) ucb_move_rate_values.push_back(ucb_context.algorithm->value(x));
                for (auto &x : oracle_context.move_rates) oracle_move_rate_values.push_back(oracle_context.algorithm->value(x));
                for (auto &x : training_context.move_rates) training_move_rate_values.push_back(training_context.algorithm->value(x));
            }
            auto pair = common::model_helpers::cost_function(session, training_move_rate_values, ucb_move_rate_values, oracle_move_rate_values);
            std::cout << "loss: " << pair.first << std::endl;
            std::cout << "results: " << std::endl;
            for (int i = 0; i < ucb_move_rate_values.size(); i++) {
                std::cout << i << ":" << std::endl << ucb_move_rate_values[i] << std::endl << oracle_move_rate_values[i] << std::endl << training_move_rate_values[i] << std::endl << pair.second[i] << std::endl;
            }
            common::model_helpers::zero_model_gradient_accumulators(session);
            training_computation_graph.accumulate_model_gradients(backprop_offset, training_move_rates, pair.second);
            common::model_helpers::apply_gradients(session);

            backprop_offset = training_computation_graph_.nodes.size();

            // store model
            if (common::model_helpers::global_step(session) % store_model_each == 0) {
                common::model_helpers::store_model(session, graph_name);
            }
        }
    }

    // close session
    TF_CHECK_OK(session->Close());

    return 0;
}