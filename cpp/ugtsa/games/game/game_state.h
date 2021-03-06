#pragma once

#include "third_party/eigen3/Eigen/Core"

#include <random>
#include <string>
#include <iostream>

namespace games {
namespace game {

class GameState {
public:
    int player;
    int player_count;
    std::default_random_engine generator;

    GameState(int player, int player_count) : player(player), player_count(player_count), generator(rand()) {}

    virtual int move_count() const = 0;
    virtual void apply_move(int index) = 0;
    virtual void undo_move() = 0;

    virtual bool is_final() const = 0;
    virtual Eigen::VectorXf payoff() const = 0;

    virtual Eigen::MatrixXf matrix() const = 0;
    virtual Eigen::VectorXf statistic() const = 0;
    virtual Eigen::VectorXf update_statistic() = 0;

    Eigen::VectorXf random_playout_payoff();
    void move_to_random_state();

    virtual GameState *copy() const = 0;
    virtual void serialize(std::ostream& stream) const = 0;

    friend std::ostream& operator<<(std::ostream& stream, const GameState& game_state);
};

std::ostream& operator<<(std::ostream& stream, const GameState& game_state);

}
}
