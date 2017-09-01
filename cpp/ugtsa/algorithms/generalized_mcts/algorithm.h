#pragma once

#include "ugtsa/algorithms/algorithm/algorithm.h"

#include <memory>

namespace algorithms {
namespace generalized_mcts {

class Algorithm : public algorithms::algorithm::Algorithm {
private:
    enum Direction {
        UP,
        DOWN
    };

    struct Worker {
        int node;
        Direction direction;
        std::unique_ptr<games::game::GameState> game_state;
        int update;
    };

    struct Node {
        int number_of_visits;
        int parent;
        int children[2];
        int statistic;
        int move_rate_cache;
    };

    std::vector<Worker> workers;
    std::vector<Node> tree;
    int grow_factor;
    std::vector<int> removed_root_moves;
    std::default_random_engine generator;

public:
    Algorithm(games::game::GameState *game_state, int worker_count, int grow_factor, std::vector<int> removed_root_moves);

    void improve();
    std::vector<int> move_rates();
    virtual Eigen::VectorXf value(int rate) = 0;

    virtual void serialize(std::ostream& stream) const;

    friend std::ostream& operator<<(std::ostream& stream, const Direction& direction);
    friend std::ostream& operator<<(std::ostream& stream, const Worker& worker);
    friend std::ostream& operator<<(std::ostream& stream, const Node& node);
    friend std::ostream& operator<<(std::ostream& stream, const Algorithm& algorithm);

protected:
    virtual int empty_statistic(games::game::GameState *game_state) = 0;
    virtual int move_rate(int parent_statistic, int child_statistic) = 0;
    virtual int game_state_as_update(games::game::GameState *game_state) = 0;
    virtual int updated_statistic(int statistic, std::vector<int> updates) = 0;
    virtual int updated_update(int update, int statistic) = 0;
    virtual void run_batch() = 0;

private:
    void down_move_case(std::vector<int> &group);
    void down_expand_case(std::vector<int> &group, std::vector<int> &empty_statistics);
    void down_leaf_case(std::vector<int> &group, std::vector<int> &game_state_as_updates);
    void up_case(std::vector<int> &group, int updated_statistic, std::vector<int> &updated_updates);
};

std::ostream& operator<<(std::ostream& stream, const Algorithm::Direction& direction);
std::ostream& operator<<(std::ostream& stream, const Algorithm::Worker& worker);
std::ostream& operator<<(std::ostream& stream, const Algorithm::Node& node);
std::ostream& operator<<(std::ostream& stream, const Algorithm& algorithm);

}
}
