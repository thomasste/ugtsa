#include "tensorflow/core/public/session.h"

#include <thread>
#include "ugtsa/common.h"
#include "ugtsa/algorithms/ucb_mcts/algorithm.h"
#include "ugtsa/algorithms/computation_graph_mcts/algorithm.h"
#include "ugtsa/computation_graphs/basic_computation_graph/computation_graph.h"
#include "ugtsa/games/omringa/game_state.h"

struct ThreadContext {
    tensorflow::Session* session;
    std::unique_ptr<computation_graphs::computation_graph::ComputationGraph> computation_graph;
    std::unique_ptr<games::game::GameState> game_state;
    std::unique_ptr<algorithms::algorithm::Algorithm> algorithm;
};

ThreadContext thread_contexts[100];

void training_thread(int i) {
    std::cout << "training " << i << std::endl;
    auto &context = thread_contexts[i];
    for (int i = 0; i < 10000; i++) {
        context.algorithm->improve();
    }

    common::zero_model_gradient_accumulators(context.session);
    context.computation_graph->accumulate_model_gradients(0, {}, {});
    common::apply_gradients(context.session);
}

void oracle_thread(int i) {
    std::cout << "oracle " << i << std::endl;
    auto &context = thread_contexts[i];
    for (int i = 0; i < 10000; i++) {
        context.algorithm->improve();
    }
}

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
    int thread_count = std::atoi(argv[11]);

    tensorflow::SessionOptions session_options = tensorflow::SessionOptions();

    std::vector<std::thread> threads;
    for (int i = 0; i < thread_count; i++) {
        // game state
        auto game_state = games::omringa::GameState();
        game_state.move_to_random_state();
        if (game_state.is_final()) {
            game_state.undo_move();
        }

        {
            auto &context = thread_contexts[threads.size()];
            TF_CHECK_OK(tensorflow::NewSession(session_options, &context.session));
            common::load_model(context.session, graph_name);

            context.computation_graph = std::unique_ptr<computation_graphs::computation_graph::ComputationGraph>(new computation_graphs::basic_computation_graph::ComputationGraph(context.session, "training:0", true));
            context.game_state = std::unique_ptr<games::game::GameState>(game_state.copy());
            auto transformations = common::create_transformations(
                context.computation_graph.get(), player_count, worker_count, statistic_size, update_size, game_state_board_shape,
                game_state_statistic_size, update_statistic_size, seed_size);
            context.algorithm = std::unique_ptr<algorithms::algorithm::Algorithm>(new algorithms::computation_graph_mcts::Algorithm(
                context.game_state.get(), worker_count, 5, {}, context.computation_graph.get(), transformations[0], transformations[1],
                transformations[2], transformations[3], transformations[4]));
            threads.push_back(std::thread(training_thread, threads.size()));
        }

        {
            auto &context = thread_contexts[threads.size()];
            TF_CHECK_OK(tensorflow::NewSession(session_options, &context.session));
            common::load_model(context.session, graph_name);

            context.computation_graph = std::unique_ptr<computation_graphs::computation_graph::ComputationGraph>(new computation_graphs::basic_computation_graph::ComputationGraph(context.session, "training:0", false));
            context.game_state = std::unique_ptr<games::game::GameState>(game_state.copy());
            auto transformations = common::create_transformations(
                context.computation_graph.get(), player_count, worker_count, statistic_size, update_size, game_state_board_shape,
                game_state_statistic_size, update_statistic_size, seed_size);
            context.algorithm = std::unique_ptr<algorithms::algorithm::Algorithm>(new algorithms::computation_graph_mcts::Algorithm(
                context.game_state.get(), worker_count, 5, {}, context.computation_graph.get(), transformations[0], transformations[1],
                transformations[2], transformations[3], transformations[4]));
            threads.push_back(std::thread(oracle_thread, threads.size()));
        }
    }

    for (int i = 0; i < threads.size(); i++) {
        auto &thread = threads[i];
        auto &thread_context = thread_contexts[i];
        thread.join();
        TF_CHECK_OK(thread_context.session->Close());
    }

    return 0;
}