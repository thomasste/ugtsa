#include <iostream>
#include "ugtsa/games/omringa/game_state.h"
#include "ugtsa/algorithms/ucb_mcts/algorithm.h"

using namespace std;

int main()
{
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
            for (int i = 0; i < 50012; i++) {
                a.improve();
            }
            std::cout << a << std::endl;
            gs.apply_move(a.best_move());
        }
    }

    std::cout << gs.payoff() << std::endl;

    cout << "Hello World!" << endl;
    return 0;
}
