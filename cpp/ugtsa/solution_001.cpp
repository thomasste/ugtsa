#include "tensorflow/core/public/session.h"

#include "ugtsa/common.h"
#include "ugtsa/algorithms/ucb_mcts/algorithm.h"
#include "ugtsa/algorithms/computation_graph_mcts/algorithm.h"
#include "ugtsa/computation_graphs/basic_computation_graph/computation_graph.h"
#include "ugtsa/games/omringa/game_state.h"

int main(int argc, char **argv) {
    // omringa__default__big_vertical_lstm_model_builder__10 2 10 300 300 7 7 2 2 30
    // omringa__default__small_vertical_lstm_model_builder__10 2 10 150 150 7 7 2 2 30
    std::string graph_name = argv[1];
    int player_count = std::atoi(argv[2]);
    int worker_count = std::atoi(argv[3]);
    int statistic_size = std::atoi(argv[4]);
    int update_size = std::atoi(argv[5]);
    auto game_state_board_shape = std::vector<int>({std::atoi(argv[6]), std::atoi(argv[7])});
    int game_state_statistic_size = std::atoi(argv[8]);
    int update_statistic_size = std::atoi(argv[9]);
    int seed_size = std::atoi(argv[10]);

    // create session
    tensorflow::Session* session;
    tensorflow::SessionOptions session_options = tensorflow::SessionOptions();
    TF_CHECK_OK(tensorflow::NewSession(session_options, &session));

    // load graph
    common::load_model(session, graph_name);

    for (int i = 0; i < 1; i++) {
        // game state
        auto game_state = games::omringa::GameState();
        // game_state.move_to_random_state();
        // if (game_state.is_final()) {
        //     game_state.undo_move();
        // }

        auto ucb_game_state = std::unique_ptr<games::game::GameState>(game_state.copy());
        auto ugtsa_game_state = std::unique_ptr<games::game::GameState>(game_state.copy());

        // create ucb algorithm
        auto ucb_algorithm = algorithms::ucb_mcts::Algorithm(ucb_game_state.get(), 10, 5, {}, std::sqrt(2.));

        // create ugtsa algorithm
        auto computation_graph = computation_graphs::basic_computation_graph::ComputationGraph(session, "training:0", true);
        auto transformations = common::create_transformations(
            &computation_graph, player_count, worker_count, statistic_size, update_size, game_state_board_shape,
            game_state_statistic_size, update_statistic_size, seed_size);
        auto ugtsa_algorithm = algorithms::computation_graph_mcts::Algorithm(
            ugtsa_game_state.get(), worker_count, 5, {}, &computation_graph, transformations[0], transformations[1],
            transformations[2], transformations[3], transformations[4]);

        for (int i = 0; i < 500; i++) {
            ucb_algorithm.improve();
        }

        for (int i = 0; i < 500; i++) {
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

        auto pair = common::cost_function(session, ugtsa_move_rate_values, ucb_move_rate_values, ucb_move_rate_values);
        std::cout << "loss: " << pair.first << std::endl;
        std::cout << "gradients: " << std::endl;
        for (auto & x : pair.second) {
            std::cout << x << std::endl;
        }

        common::zero_model_gradient_accumulators(session);
        computation_graph.accumulate_model_gradients(0, ugtsa_move_rates, pair.second);

        // store model
        common::store_model(session, graph_name + "2");
    }

    // close session
    TF_CHECK_OK(session->Close());

    return 0;
}