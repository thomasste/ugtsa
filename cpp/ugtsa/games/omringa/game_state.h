#pragma once

#include "ugtsa/games/game/game_state.h"

namespace games {
namespace omringa {

class GameState : public games::game::GameState {
    struct Position {
        int x;
        int y;
    };

    enum State {
        BET,
        NATURE,
        PLACE
    };

    struct Move {
        State state;
        int player;
        Position position;
        int value;
        int index;
    };

    int board_size;
    int group_penalty;
    int min_bet;
    int max_bet;

    State state;
    int bets[2];
    int chosen_player;
    Eigen::MatrixXi board;
    std::vector<Position> empty_positions;
    std::vector<Move> move_history;

public:
    GameState(int board_size=7, int group_penalty=-5, int min_bet=0, int max_bet=9);
    GameState(const GameState &game_state);

    int move_count() const;
    void apply_move(int index);
    void undo_move();

    bool is_final() const;
    Eigen::VectorXf payoff() const;

    Eigen::MatrixXf matrix() const;
    Eigen::VectorXf statistic() const;
    Eigen::VectorXf update_statistic();

    GameState *copy() const;
    void serialize(std::ostream& stream) const;

private:
    Move get_move(int index) const;
    int count_groups(int player) const;

    friend std::ostream& operator<<(std::ostream& stream, const Position& position);
    friend std::ostream& operator<<(std::ostream& stream, const State& state);
    friend std::ostream& operator<<(std::ostream& stream, const Move& move);
    friend std::ostream& operator<<(std::ostream& stream, const GameState& game_state);
};

std::ostream& operator<<(std::ostream& stream, const GameState::Position& position);
std::ostream& operator<<(std::ostream& stream, const GameState::State& state);
std::ostream& operator<<(std::ostream& stream, const GameState::Move& move);
std::ostream& operator<<(std::ostream& stream, const GameState& game_state);

}
}
