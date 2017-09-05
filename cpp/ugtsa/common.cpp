#include "ugtsa/common.h"

namespace common {

void load_model(tensorflow::Session* session, std::string graph_name) {
    auto graph_def = tensorflow::GraphDef();
    TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), "graphs/" + graph_name + ".pb", &graph_def));
    TF_CHECK_OK(session->Create(graph_def));

    auto model_path = tensorflow::Tensor(tensorflow::DT_STRING, tensorflow::TensorShape({1, 1}));
    model_path.matrix<std::string>()(0, 0) = "models/" + graph_name;
    TF_CHECK_OK(session->Run({{"save/Const:0", model_path}}, {}, {"save/restore_all"}, nullptr));
}

void store_model(tensorflow::Session* session, std::string graph_name) {
    auto model_path = tensorflow::Tensor(tensorflow::DT_STRING, tensorflow::TensorShape({1, 1}));
    model_path.matrix<std::string>()(0, 0) = "models/" + graph_name;
    TF_CHECK_OK(session->Run({{"save/Const", model_path}}, {}, {"save/control_dependency:0"}, nullptr));
}

std::vector<int> create_transformations(
        computation_graphs::computation_graph::ComputationGraph *computation_graph,
        int player_count,
        int worker_count,
        int statistic_size,
        int update_size,
        std::vector<int> game_state_board_shape,
        int game_state_statistic_size,
        int update_statistic_size,
        int seed_size) {
    // std::cout << "player_count " << player_count << std::endl
    //           << "worker_count " << worker_count << std::endl
    //           << "statistic_size " << statistic_size << std::endl
    //           << "update_size " << update_size << std::endl
    //           << "game_state_board_shape " << game_state_board_shape[0] << " " << game_state_board_shape[1] << std::endl
    //           << "game_state_statistic_size " << game_state_statistic_size << std::endl
    //           << "update_statistic_size " << update_statistic_size << std::endl
    //           << "seed_size " << seed_size << std::endl;

    auto empty_statistic = computation_graph->transformation(
        "empty_statistic/seed:0",
        seed_size,
        {"empty_statistic/game_state_board:0", "empty_statistic/game_state_statistic:0"},
        {{game_state_board_shape}, {game_state_statistic_size}},
        {tensorflow::DataType::DT_FLOAT, tensorflow::DataType::DT_FLOAT},
        {"", ""},
        "empty_statistic/output:0",
        {statistic_size},
        tensorflow::DataType::DT_FLOAT,
        "empty_statistic/output_gradient:0",
        "empty_statistic/update_model_gradient_accumulators");
    auto move_rate = computation_graph->transformation(
        "move_rate/seed:0",
        seed_size,
        {"move_rate/parent_statistic:0", "move_rate/child_statistic:0"},
        {{statistic_size}, {statistic_size}},
        {tensorflow::DataType::DT_FLOAT, tensorflow::DataType::DT_FLOAT},
        {"move_rate/parent_statistic_gradient:0", "move_rate/child_statistic_gradient:0"},
        "move_rate/output:0",
        {player_count},
        tensorflow::DataType::DT_FLOAT,
        "move_rate/output_gradient:0",
        "move_rate/update_model_gradient_accumulators");
    auto game_state_as_update = computation_graph->transformation(
        "game_state_as_update/seed:0",
        seed_size,
        {"game_state_as_update/update_statistic:0"},
        {{update_statistic_size}},
        {tensorflow::DataType::DT_FLOAT},
        {""},
        "game_state_as_update/output:0",
        {update_size},
        tensorflow::DataType::DT_FLOAT,
        "game_state_as_update/output_gradient:0",
        "game_state_as_update/update_model_gradient_accumulators");
    auto updated_statistic = computation_graph->transformation(
        "updated_statistic/seed:0",
        seed_size,
        {"updated_statistic/statistic:0", "updated_statistic/update_count:0", "updated_statistic/updates:0"},
        {{statistic_size}, {}, {worker_count * update_size}},
        {tensorflow::DataType::DT_FLOAT, tensorflow::DataType::DT_INT32, tensorflow::DataType::DT_FLOAT},
        {"updated_statistic/statistic_gradient:0", "", "updated_statistic/updates_gradient:0"},
        "updated_statistic/output:0",
        {statistic_size},
        tensorflow::DataType::DT_FLOAT,
        "updated_statistic/output_gradient:0",
        "updated_statistic/update_model_gradient_accumulators");
    auto updated_update = computation_graph->transformation(
        "updated_update/seed:0",
        seed_size,
        {"updated_update/update:0", "updated_update/statistic:0"},
        {{update_size}, {statistic_size}},
        {tensorflow::DataType::DT_FLOAT, tensorflow::DataType::DT_FLOAT},
        {"updated_update/update_gradient:0", "updated_update/statistic_gradient:0"},
        "updated_update/output:0",
        {update_size},
        tensorflow::DataType::DT_FLOAT,
        "updated_update/output_gradient:0",
        "updated_update/update_model_gradient_accumulators");
    return {empty_statistic, move_rate, game_state_as_update, updated_statistic, updated_update};
}

void zero_model_gradient_accumulators(tensorflow::Session* session) {
    std::vector<std::string> run_fetches;

    for (std::string transformation : {"empty_statistic", "move_rate", "game_state_as_update", "updated_statistic", "updated_update", "cost_function"}) {
        run_fetches.push_back(transformation + "/zero_model_gradient_accumulators");
    }

    session->Run({}, {}, run_fetches, nullptr);
}

std::pair<float, std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>>> cost_function(
        tensorflow::Session* session,
        std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> move_rates,
        std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> ucb_move_rates,
        std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> ugtsa_move_rates) {
    // feed_dict
    std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;

    std::vector<tensorflow::Tensor> tensors;
    for (auto &vector : {move_rates, ucb_move_rates, ugtsa_move_rates}) {
        auto tensor = tensorflow::Tensor(tensorflow::DataType::DT_FLOAT, {vector.size(), vector[0].size()});
        for (int i = 0; i < vector.size(); i++) {
            auto &v = vector[i];
            for (int j = 0; j < v.size(); j++) tensor.matrix<float>()(i, j) = v(j);
        }
        tensors.push_back(tensor);
    }

    feed_dict.push_back(std::make_pair("cost_function/move_rates:0", tensors[0]));
    feed_dict.push_back(std::make_pair("cost_function/ucb_move_rates:0", tensors[1]));
    feed_dict.push_back(std::make_pair("cost_function/ugtsa_move_rates:0", tensors[2]));

    // run
    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(session->Run(feed_dict, {"cost_function/output:0", "cost_function/move_rates_gradient:0"}, {}, &outputs));

    float loss = outputs[0].flat<float>()(0);

    std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> gradients;
    for (int i = 0; i < move_rates.size(); i++) {
        Eigen::VectorXf gradient(move_rates[0].size());
        for (int j = 0; j < move_rates[0].size(); j++) gradient(j) = outputs[1].matrix<float>()(i, j);
        gradients.push_back(gradient);
    }

    return std::make_pair(loss, gradients);
}

void apply_gradients(tensorflow::Session* session) {
    session->Run({}, {}, {"apply_gradients/apply_gradients"}, nullptr);
}

}