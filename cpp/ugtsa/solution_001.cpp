#include "tensorflow/core/public/session.h"

#include "ugtsa/algorithms/ucb_mcts/algorithm.h"
#include "ugtsa/algorithms/computation_graph_mcts/algorithm.h"
#include "ugtsa/computation_graphs/basic_computation_graph/computation_graph.h"
#include "ugtsa/games/omringa/game_state.h"

int main(int argc, char **argv) {
    // create session
    tensorflow::Session* session;
    tensorflow::SessionOptions session_options = tensorflow::SessionOptions();
    TF_CHECK_OK(tensorflow::NewSession(session_options, &session));

    // load graph
    auto graph_def = tensorflow::GraphDef();
    TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), "graphs/omringa__default__big_vertical_lstm_model_builder__10.pb", &graph_def));
    TF_CHECK_OK(session->Create(graph_def));

    // load model
    auto model_path = tensorflow::Tensor(tensorflow::DT_STRING, tensorflow::TensorShape({1, 1}));
    model_path.matrix<std::string>()(0, 0) = "models/omringa__default__big_vertical_lstm_model_builder__10";
    TF_CHECK_OK(session->Run({{"save/Const:0", model_path}}, {}, {"save/restore_all"}, nullptr));

    // game state
    auto game_state = games::omringa::GameState();
    auto ucb_game_state = games::omringa::GameState(); // = std::unique_ptr<games::game::GameState>(game_state.copy());
    auto ugtsa_game_state = games::omringa::GameState(); // = std::unique_ptr<games::game::GameState>(game_state.copy());

    // games::game::GameState *game_state, int worker_count, int grow_factor, std::vector<int> removed_root_moves, float exploration_factor
    auto ucb_algorithm = algorithms::ucb_mcts::Algorithm(&ucb_game_state, 10, 5, {}, std::sqrt(2.));

    // TODO
    int player_count = 2;
    int worker_count = 10;
    int statistic_size = 300;
    int update_size = 300;
    auto game_state_board_shape = std::vector<int>({7, 7});
    int game_state_statistic_size = 2;
    int update_statistic_size = 2;
    int seed_size=30;

    auto computation_graph = computation_graphs::basic_computation_graph::ComputationGraph(session, "training:0", true);
    auto empty_statistic = computation_graph.transformation(
        "empty_statistic/seed:0",
        seed_size,
        {"empty_statistic/game_state_board:0", "empty_statistic/game_state_statistic:0"},
        {{game_state_board_shape}, {game_state_statistic_size}},
        {tensorflow::DataType::DT_FLOAT, tensorflow::DataType::DT_FLOAT},
        {"", ""},
        "empty_statistic/output:0",
        {statistic_size},
        tensorflow::DataType::DT_FLOAT,
        "empty_statistic/output_gradient:0");

    auto move_rate = computation_graph.transformation(
        "move_rate/seed:0",
        seed_size,
        {"move_rate/parent_statistic:0", "move_rate/child_statistic:0"},
        {{statistic_size}, {statistic_size}},
        {tensorflow::DataType::DT_FLOAT, tensorflow::DataType::DT_FLOAT},
        {"move_rate/parent_statistic_gradient:0", "move_rate/child_statistic_gradient:0"},
        "move_rate/output:0",
        {player_count},
        tensorflow::DataType::DT_FLOAT,
        "move_rate/output_gradient:0");
    auto game_state_as_update = computation_graph.transformation(
        "game_state_as_update/seed:0",
        seed_size,
        {"game_state_as_update/update_statistic:0"},
        {{update_statistic_size}},
        {tensorflow::DataType::DT_FLOAT},
        {""},
        "game_state_as_update/output:0",
        {update_size},
        tensorflow::DataType::DT_FLOAT,
        "game_state_as_update/output_gradient:0");
    auto updated_statistic = computation_graph.transformation(
        "updated_statistic/seed:0",
        seed_size,
        {"updated_statistic/statistic:0", "updated_statistic/update_count:0", "updated_statistic/updates:0"},
        {{statistic_size}, {}, {worker_count * update_size}},
        {tensorflow::DataType::DT_FLOAT, tensorflow::DataType::DT_INT32, tensorflow::DataType::DT_FLOAT},
        {"updated_statistic/statistic_gradient:0", "", "updated_statistic/updates_gradient:0"},
        "updated_statistic/output:0",
        {statistic_size},
        tensorflow::DataType::DT_FLOAT,
        "updated_statistic/output_gradient:0");
    auto updated_update = computation_graph.transformation(
        "updated_update/seed:0",
        seed_size,
        {"updated_update/update:0", "updated_update/statistic:0"},
        {{update_size}, {statistic_size}},
        {tensorflow::DataType::DT_FLOAT, tensorflow::DataType::DT_FLOAT},
        {"updated_update/update_gradient:0", "updated_update/statistic_gradient:0"},
        "updated_update/output:0",
        {update_size},
        tensorflow::DataType::DT_FLOAT,
        "updated_update/output_gradient:0");
    // games::game::GameState *game_state, int worker_count, int grow_factor, std::vector<int> removed_root_moves,
    // computation_graphs::computation_graph::ComputationGraph *computation_graph, int empty_statistic, int move_rate, int game_state_as_update, int updated_statistic, int updated_update
    auto ugtsa_algorithm = algorithms::computation_graph_mcts::Algorithm(
        &ugtsa_game_state, 10, 5, {}, &computation_graph, empty_statistic, move_rate,
        game_state_as_update, updated_statistic, updated_update);

    for (int i = 0; i < 1000; i++) {
        ugtsa_algorithm.improve();
    }

    std::cout << ugtsa_algorithm << std::endl;

    // close session
    TF_CHECK_OK(session->Close());

    std::cout << "DUPA" << std::endl;

    return 0;
}