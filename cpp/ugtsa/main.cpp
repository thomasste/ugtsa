#include "tensorflow/core/public/session.h"

#include "ugtsa/algorithms/ucb_mcts/algorithm.h"
#include "ugtsa/computation_graphs/basic_computation_graph/computation_graph.h"
#include "ugtsa/games/omringa/game_state.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    auto gs = games::omringa::GameState();

    std::default_random_engine generator;
//    cout << gs << std::endl;
//    cout << gs.random_playout_payoff() << endl;
//    cout << gs.move_count() << endl;

    while (!gs.is_final()) {
        std::cout << gs << std::endl;
        if (gs.player == -1) {
            gs.apply_move(std::uniform_int_distribution<int>(0, gs.move_count() - 1)(generator));
        } else {
            auto a = algorithms::ucb_mcts::Algorithm(&gs, 10, 5, std::vector<int>(), std::sqrt(2.));
            for (int i = 0; i < 500000; i++) {
                a.improve();
            }
            //std::cout << a << std::endl;
            gs.apply_move(a.best_move());
        }
    }

    std::cout << gs.payoff() << std::endl;

    std::cout << "Hello World!" << std::endl;
    return 0;

    // tensorflow::Session* session;
    // tensorflow::SessionOptions session_options = tensorflow::SessionOptions();
    // tensorflow::Status status = tensorflow::NewSession(session_options, &session);
    // if (!status.ok()) {
    //     std::cout << status.ToString() << "\n";
    //     return 1;
    // }

    // tensorflow::GraphDef graph_def;
    // status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), "graphs/omringa__default__big_vertical_lstm_model_builder__10.pb", &graph_def);
    // if (!status.ok()) {
    //     std::cout << status.ToString() << "\n";
    //     return 1;
    // }

    // status = session->Create(graph_def);
    // if (!status.ok()) {
    //     std::cout << status.ToString() << "\n";
    //     return 1;
    // }

    // tensorflow::Tensor model_path(tensorflow::DT_STRING, tensorflow::TensorShape({1, 1}));
    // model_path.matrix<std::string>()(0, 0) = "models/omringa__default__big_vertical_lstm_model_builder__10";

    // status = session->Run({{"save/Const:0", model_path}}, {}, {"save/restore_all"}, nullptr);

    // if (!status.ok()) {
    //     std::cout << status.ToString() << "\n";
    //     return 1;
    // }

    // auto game_state = games::omringa::GameState();

    // auto board1 = game_state.matrix();
    // auto board2 = game_state.matrix();
    // auto payoff1 = game_state.random_playout_payoff();
    // auto payoff2 = game_state.random_playout_payoff();

    // std::cout << board1 << std::endl;
    // std::cout << payoff1 << std::endl;
    // std::cout << board2 << std::endl;
    // std::cout << payoff2 << std::endl;

    // tensorflow::Tensor training(tensorflow::DataType::DT_BOOL, tensorflow::TensorShape({}));
    // tensorflow::Tensor seed(tensorflow::DataType::DT_INT64, tensorflow::TensorShape({30}));
    // tensorflow::Tensor boards(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({2, 7, 7}));
    // tensorflow::Tensor payoffs(tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({2, 2}));

    // training.scalar<bool>()(0) = true;

    // for (int i = 0; i < 30; i++) {
    //     seed.tensor<long long int, 1>()(i) = i;
    // }

    // for (int i = 0; i < board1.rows(); i++) {
    //     for (int j = 0; j < board1.cols(); j++) {
    //         boards.tensor<float, 3>()(0, i, j) = board1(i, j);
    //     }
    // }

    // for (int i = 0; i < board2.rows(); i++) {
    //     for (int j = 0; j < board2.cols(); j++) {
    //         boards.tensor<float, 3>()(1, i, j) = board2(i, j);
    //     }
    // }

    // for (int i = 0; i < payoff1.size(); i++) {
    //     payoffs.tensor<float, 2>()(0, i) = payoff1(i);
    // }

    // for (int i = 0; i < payoff2.size(); i++) {
    //     payoffs.tensor<float, 2>()(1, i) = payoff2(i);
    // }

    // std::cout << boards.tensor<float, 3>() << std::endl;
    // std::cout << boards.shape().dim_size(0) << " " << boards.shape().dim_size(1) << " " << boards.shape().dim_size(2) << std::endl;
    // std::cout << payoffs.tensor<float, 2>() << std::endl;
    // std::cout << payoffs.shape().dim_size(0) << " " << payoffs.shape().dim_size(1) << " " << std::endl;

    // std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
    // inputs.push_back(std::make_pair("training:0", training));
    // inputs.push_back(std::make_pair("empty_statistic/seed:0", seed));
    // inputs.push_back(std::make_pair("empty_statistic/game_state_board:0", boards));
    // inputs.push_back(std::make_pair("empty_statistic/game_state_statistic:0", payoffs));
    // std::vector<tensorflow::Tensor> outputs;

    // status = session->Run(inputs, {"empty_statistic/output:0"}, {}, &outputs);
    // if (!status.ok()) {
    //     std::cout << status.ToString() << std::endl;
    // }

    // std::cout << outputs.size() << std::endl;

    // std::cout << outputs[0].tensor<float, 2>() << std::endl;

    // status = session->Run({{"save/Const", model_path}}, {}, {"save/control_dependency:0"}, nullptr);
    // if (!status.ok()) {
    //     std::cout << status.ToString() << std::endl;
    // }

    // computation_graphs::basic_computation_graph::ComputationGraph computation_graph(session, "training:0", true);
    // auto t = computation_graph.transformation(
    //     "empty_statistic/seed:0",
    //     30,
    //     {"empty_statistic/game_state_board:0", "empty_statistic/game_state_statistic:0"},
    //     {{7, 7}, {2}},
    //     {tensorflow::DataType::DT_FLOAT, tensorflow::DataType::DT_FLOAT},
    //     {"", ""},
    //     "empty_statistic/output:0",
    //     {300},
    //     tensorflow::DataType::DT_FLOAT,
    //     "empty_statistic/output_gradient:0");
    // auto b1 = computation_graph.matrix(board1);
    // auto b2 = computation_graph.matrix(board2);
    // auto p1 = computation_graph.matrix(payoff1);
    // auto p2 = computation_graph.matrix(payoff2);
    // auto r1 = computation_graph.transformation_run(t, {{b1}, {p1}});
    // auto r2 = computation_graph.transformation_run(t, {{b2}, {p2}});
    // computation_graph.run_batch();

    // status = session->Close();
    // if (!status.ok()) {
    //     std::cout << status.ToString() << std::endl;
    // }

    // return 0;
}
