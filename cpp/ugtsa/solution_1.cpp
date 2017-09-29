#include "tensorflow/core/public/session.h"

#include "ugtsa/algorithms/ucb_mcts/algorithm.h"
#include "ugtsa/algorithms/computation_graph_mcts/algorithm.h"
#include "ugtsa/common/model_helpers.h"
#include "ugtsa/computation_graphs/basic_computation_graph/computation_graph.h"
#include "ugtsa/games/omringa/game_state.h"

int main(int argc, char **argv) {
    srand(time(NULL));

    // omringa__default__big_vertical_lstm_model_builder__10 5 80. 500 500 1 1
    // omringa__default__small_vertical_lstm_model_builder__10 5 80. 500 500 1 1
    std::string graph_name = argv[1];
    int grow_factor = std::atoi(argv[2]);
    float move_choice_factor = std::atof(argv[3]);
    int ucb_strength = std::atoi(argv[4]);
    int ugtsa_strength = std::atoi(argv[5]);
    int number_of_iterations = std::atoi(argv[6]);
    int store_model_each = std::atoi(argv[7]);

    // create session
    tensorflow::Session* session;
    tensorflow::SessionOptions session_options = tensorflow::SessionOptions();
    TF_CHECK_OK(tensorflow::NewSession(session_options, &session));

    // load graph
    common::model_helpers::load_model(session, graph_name);
    int worker_count = common::model_helpers::worker_count(session);

    for (int i = 0; i < number_of_iterations; i++) {
        std::cout << "global step " << common::model_helpers::global_step(session) << std::endl;

        // game state
        auto game_state = games::omringa::GameState();
        game_state.move_to_random_state();
        if (game_state.is_final()) {
            game_state.undo_move();
        }

        auto ucb_game_state = std::unique_ptr<games::game::GameState>(game_state.copy());
        auto ugtsa_game_state = std::unique_ptr<games::game::GameState>(game_state.copy());

        // create ucb algorithm
        auto ucb_algorithm = algorithms::ucb_mcts::Algorithm(
            ucb_game_state.get(), worker_count, grow_factor, move_choice_factor, {}, std::sqrt(2.));

        // create ugtsa algorithm
        auto computation_graph = computation_graphs::basic_computation_graph::ComputationGraph(
            session, "training:0", true);
        auto transformations = common::model_helpers::create_transformations(session, &computation_graph);
        auto ugtsa_algorithm = algorithms::computation_graph_mcts::Algorithm(
            ugtsa_game_state.get(), worker_count, grow_factor, move_choice_factor, {}, &computation_graph,
            transformations[0], transformations[1], transformations[2], transformations[3], transformations[4]);

        for (int i = 0; i < ucb_strength; i++) {
            ucb_algorithm.improve();
        }

        for (int i = 0; i < ugtsa_strength; i++) {
            ugtsa_algorithm.improve();
        }

        auto ucb_move_rates = ucb_algorithm.move_rates();
        auto ugtsa_move_rates = ugtsa_algorithm.move_rates();
        std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> ucb_move_rate_values;
        std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> ugtsa_move_rate_values;
        for (auto &x : ucb_move_rates) ucb_move_rate_values.push_back(ucb_algorithm.value(x));
        for (auto &x : ugtsa_move_rates) ugtsa_move_rate_values.push_back(ugtsa_algorithm.value(x));

        std::cout << "ucb_move_rate_values" << std::endl;
        for (auto &x : ucb_move_rate_values) {
            std::cout << x << std::endl;
        }
        std::cout << "ugtsa_move_rate_values" << std::endl;
        for (auto &x : ugtsa_move_rate_values) {
            std::cout << x << std::endl;
        }

        auto pair = common::model_helpers::cost_function(session, ugtsa_move_rate_values, ucb_move_rate_values, ucb_move_rate_values);
        std::cout << "loss: " << pair.first << std::endl;
        std::cout << "gradients: " << std::endl;
        for (auto & x : pair.second) {
            std::cout << x << std::endl;
        }

        common::model_helpers::zero_model_gradient_accumulators(session);
        computation_graph.accumulate_model_gradients(0, ugtsa_move_rates, pair.second);
        common::model_helpers::apply_gradients(session);

        // store model
        if (common::model_helpers::global_step(session) % store_model_each == 0) {
            common::model_helpers::store_model(session, graph_name);
        }
    }

    // close session
    TF_CHECK_OK(session->Close());

    return 0;
}