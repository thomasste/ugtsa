#include "tensorflow/core/public/session.h"

#include <thread>
#include "ugtsa/common.h"
#include "ugtsa/algorithms/ucb_mcts/algorithm.h"
#include "ugtsa/algorithms/computation_graph_mcts/algorithm.h"
#include "ugtsa/computation_graphs/basic_computation_graph/computation_graph.h"
#include "ugtsa/computation_graphs/synchronized_computation_graph/computation_graph.h"
#include "ugtsa/games/omringa/game_state.h"

class Barrier {
public:
    explicit Barrier(std::size_t iCount) :
      mThreshold(iCount),
      mCount(iCount),
      mGeneration(0) {
    }

    void wait() {
        std::unique_lock<std::mutex> lLock{mMutex};
        auto lGen = mGeneration;
        if (!--mCount) {
            mGeneration++;
            mCount = mThreshold;
            mCond.notify_all();
        } else {
            mCond.wait(lLock, [this, lGen] { return lGen != mGeneration; });
        }
    }

private:
    std::mutex mMutex;
    std::condition_variable mCond;
    std::size_t mThreshold;
    std::size_t mCount;
    std::size_t mGeneration;
};

struct ThreadContext {
    tensorflow::Session* session;
    computation_graphs::computation_graph::ComputationGraph *computation_graph;
    std::unique_ptr<games::game::GameState> game_state;
    std::unique_ptr<algorithms::algorithm::Algorithm> algorithm;
    std::shared_ptr<Barrier> barrier;
};

ThreadContext thread_contexts[100];

void training_thread(int i) {
    std::cout << "training " << i << std::endl;
    auto &context = thread_contexts[i];
    for (int i = 0; i < 10000; i++) {
        context.algorithm->improve();
    }
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
    int number_of_iterations = std::atoi(argv[12]);
    int thread_count = std::atoi(argv[11]);

    tensorflow::SessionOptions session_options = tensorflow::SessionOptions();

    tensorflow::Session *session;
    TF_CHECK_OK(tensorflow::NewSession(session_options, &session));
    int global_step = common::load_model(session, graph_name);

    computation_graphs::basic_computation_graph::ComputationGraph training_cg(session, "training:0", true);
    computation_graphs::basic_computation_graph::ComputationGraph oracle_cg(session, "training:0", false);

    computation_graphs::synchronized_computation_graph::ComputationGraph training_computation_graph(&training_cg, thread_count);
    computation_graphs::synchronized_computation_graph::ComputationGraph oracle_computation_graph(&oracle_cg, thread_count);

    auto training_transformations = common::create_transformations(
        &training_computation_graph, player_count, worker_count, statistic_size, update_size, game_state_board_shape,
        game_state_statistic_size, update_statistic_size, seed_size);

    auto oracle_transformations = common::create_transformations(
        &oracle_computation_graph, player_count, worker_count, statistic_size, update_size, game_state_board_shape,
        game_state_statistic_size, update_statistic_size, seed_size);

    for (int i = 0; i < number_of_iterations; i++) {
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
                context.session = session;
                context.computation_graph = &training_computation_graph;
                context.game_state = std::unique_ptr<games::game::GameState>(game_state.copy());
                context.algorithm = std::unique_ptr<algorithms::algorithm::Algorithm>(new algorithms::computation_graph_mcts::Algorithm(
                    context.game_state.get(), worker_count, 5, {}, context.computation_graph, training_transformations[0], training_transformations[1],
                    training_transformations[2], training_transformations[3], training_transformations[4]));
                threads.push_back(std::thread(training_thread, threads.size()));
            }

            {
                auto &context = thread_contexts[threads.size()];
                context.session = session;
                context.computation_graph = &oracle_computation_graph;
                context.game_state = std::unique_ptr<games::game::GameState>(game_state.copy());
                context.algorithm = std::unique_ptr<algorithms::algorithm::Algorithm>(new algorithms::computation_graph_mcts::Algorithm(
                    context.game_state.get(), worker_count, 5, {}, context.computation_graph, oracle_transformations[0], oracle_transformations[1],
                    oracle_transformations[2], oracle_transformations[3], oracle_transformations[4]));
                threads.push_back(std::thread(oracle_thread, threads.size()));
            }
        }

        for (int i = 0; i < threads.size(); i++) {
            auto &thread = threads[i];
            thread.join();
        }

        common::zero_model_gradient_accumulators(session);
        training_computation_graph.accumulate_model_gradients(0, {}, {});
        common::apply_gradients(session);

        global_step++;

        if (global_step % 20 == 0) {
            common::store_model(session, graph_name);
        }
    }

    TF_CHECK_OK(session->Close());

    return 0;
}