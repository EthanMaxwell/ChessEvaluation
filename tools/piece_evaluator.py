#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chess.engine
import json
import argparse
import random
import time
import sys
import asyncio
import pathlib
import logging
import math
from deap import base, creator, tools, algorithms


async def load_engine_from_cmd(cmd, debug=False):
    _, engine = await chess.engine.popen_uci(cmd.split())
    if hasattr(engine, "debug"):
        engine.debug(debug)
    return engine


def get_user_move(board):
    # Get well-formated move
    move = None
    while move is None:
        san_option = random.choice([board.san(m) for m in board.legal_moves])
        uci_option = random.choice([m.uci() for m in board.legal_moves])
        uci = input(f"Your move (e.g. {san_option} or {uci_option}): ")
        if uci in ("quit", "exit"):
            return None

        for parse in (board.parse_san, chess.Move.from_uci):
            try:
                move = parse(uci)
            except ValueError:
                pass

    # Check legality
    if move not in board.legal_moves:
        print("Illegal move.")
        return get_user_move(board)

    return move


def get_user_color():
    color = ""
    while color not in ("white", "black"):
        color = input("Do you want to be white or black? ")
    return chess.WHITE if color == "white" else chess.BLACK


def print_unicode_board(board, perspective=chess.WHITE):
    """Prints the position from a given perspective."""
    sc, ec = "\x1b[0;30;107m", "\x1b[0m"
    for r in range(8) if perspective == chess.BLACK else range(7, -1, -1):
        line = [f"{sc} {r+1}"]
        for c in range(8) if perspective == chess.WHITE else range(7, -1, -1):
            color = "\x1b[48;5;255m" if (r + c) % 2 == 1 else "\x1b[48;5;253m"
            if board.move_stack:
                if board.move_stack[-1].to_square == 8 * r + c:
                    color = "\x1b[48;5;153m"
                elif board.move_stack[-1].from_square == 8 * r + c:
                    color = "\x1b[48;5;153m"
            piece = board.piece_at(8 * r + c)
            line.append(
                color + (chess.UNICODE_PIECE_SYMBOLS[piece.symbol()] if piece else " ")
            )
        print(" " + " ".join(line) + f" {sc} {ec}")
    if perspective == chess.WHITE:
        print(f" {sc}   a b c d e f g h  {ec}\n")
    else:
        print(f" {sc}   h g f e d c b a  {ec}\n")
        
def piece_placement(pieces):
    """Places the given pieces onto FEN formate for one side of the board"""
    piece_names = ['q', 'b', 'n', 'r', 'p']
    queue =  []
    for index, count in enumerate(pieces):
        letter = piece_names[index]  # Get the corresponding letter
        queue.extend([letter] * count)
    queue.reverse()
    
    board = ["k"]
    cur_line = 0
    left = True
    
    while len(queue) > 0:
        cur_piece = queue.pop()
        
        if len(board[cur_line]) >= 8 or (cur_line == 0 and cur_piece == "p") :
            cur_line += 1
            board.append("")
        
        if left:
            board[cur_line] = cur_piece + board[cur_line]
        else:
            board[cur_line] = board[cur_line] + cur_piece
            
        left = not left
    
    def fill_empty(line):
        empty = (8 - len(line)) / 2
        start = math.ceil(empty)
        end = math.floor(empty)
        line = (str(start) if start else "") + line + (str(end) if end else "")
        return line
    
    for pos in range(len(board)):
        board[pos] = fill_empty(board[pos])
        
    return board

def make_board_fen(black, white):
    """Puts all the given pieces into complete FEN formatted board"""
    black_board = piece_placement(black)
    white_board = piece_placement(white)
    white_board.reverse()

    board = "/".join(black_board)
    
    remaining_rows = 8 - len(black_board) - len(white_board)
    if remaining_rows < 0:
        raise ValueError("white and black pieces overlap")
    elif remaining_rows == 0:
        blank_rows = "/"
    else:
        blank_rows = "/" + "/".join(["8"] * remaining_rows) + "/"
    board += blank_rows
    
    board += "/".join(white_board).upper()

    return board +  " w KQkq - 0 1"

async def get_engine_move(engine, board, limit, game_id, multipv, debug=False):
    # XBoard engine doesn't support multipv, and there python-chess doesn't support
    # getting the first PV while playing a game.
    if isinstance(engine, chess.engine.XBoardProtocol):
        play_result = await engine.play(board, limit, game=game_id)
        return play_result.move
    
    multipv = min(multipv, board.legal_moves.count())
    with await engine.analysis(
        board, limit, game=game_id, info=chess.engine.INFO_ALL, multipv=multipv or None
    ) as analysis:

        infos = [None for _ in range(multipv)]
        first = True
        async for new_info in analysis:
            # If multipv = 0 it means we don't want them at all,
            # but uci requires MultiPV to be at least 1.
            if multipv and "multipv" in new_info:
                infos[new_info["multipv"] - 1] = new_info

            # Parse optional arguments into a dict
            if debug and "string" in new_info:
                print(new_info["string"])

            if not debug and all(infos) and "score" in analysis.info:
                if not first:
                    # print('\n'*(multipv+1), end='')
                    print(f"\u001b[1A\u001b[K" * (multipv + 1), end="")
                else:
                    first = False

                info = analysis.info
                score = info["score"].relative
                score = (
                    f"Score: {score.score()}"
                    if score.score() is not None
                    else f"Mate in {score.mate()}"
                )
                print(
                    f'{score}, nodes: {info.get("nodes", "N/A")}, nps: {info.get("nps", "N/A")},'
                    f' time: {float(info.get("time", 0)):.1f}',
                    end="",
                )
                print()

                for info in infos:
                    if "pv" in info:
                        variation = board.variation_san(info["pv"][:10])
                    else:
                        variation = ""

                    if "score" in info:
                        score = info["score"].relative
                        score = (
                            math.tanh(score.score() / 600)
                            if score.score() is not None
                            else score.mate()
                        )
                        key, *val = info.get("string", "").split()
                        if key == "pv_nodes":
                            nodes = int(val[0])
                            rel = nodes / analysis.info["nodes"]
                            score_rel = f"({score:.2f}, {rel*100:.0f}%)"
                        else:
                            score_rel = f"({score:.2f})"
                    else:
                        score_rel = ""

                    # Something about N
                    print(f'{info["multipv"]}: {score_rel} {variation}')

        return analysis.info["pv"][0]


async def play(engine, board, selfplay, pvs, time_limit, debug=False, printout=False):
    if not selfplay:
        user_color = get_user_color()
    else:
        user_color = chess.WHITE

    if not board:
        board = chess.Board()

    game_id = random.random()

    while not board.is_game_over():
        if printout: print_unicode_board(board, perspective=user_color)
        if not selfplay and user_color == board.turn:
            move = get_user_move(board)
            if move is None:    
                return
        else:
            move = await get_engine_move(
                engine, board, time_limit, game_id, pvs, debug=debug
            )
            if printout: print(f" My move: {board.san(move)}")
        board.push(move)

    # Print status
    if printout:
        print_unicode_board(board, perspective=user_color)
        print("Result:", board.result())
        
    return board.outcome().winner

async def run_ea(engine):
    toolbox = setup_toolbox(engine)
    
    # Create initial population
    population = toolbox.population(n=50)
    
    # Evaluate initial population
    loop = asyncio.get_event_loop()
    fitnesses = await asyncio.gather(*(loop.run_until_complete(toolbox.evaluate(ind)) for ind in population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Run the evolution
    for gen in range(40):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = await asyncio.gather(*(toolbox.evaluate(ind) for ind in invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace the old population by the offspring
        population[:] = offspring
        print(population[0])
    
    # Return the best individual
    return tools.selBest(population, 1)[0]

def setup_toolbox(engine):
    # Define the individual and population
    toolbox = base.Toolbox()
    # Define attribute generators for different ranges
    def attr_int_queen():
        return random.randint(0, 2)
    def attr_int_bnr():
        return random.randint(0, 4)

    def attr_int_pawn():
        return random.randint(0, 15)

    # Define the individual and population
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (attr_int_queen, attr_int_bnr, attr_int_bnr, attr_int_bnr, attr_int_pawn), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=[0, 0, 0, 0, 0], up=[4, 10, 10, 10, 25], indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", async_fitness_function, engine=engine)
    
    return toolbox

async def async_fitness_function(individual, engine):
    return sum(individual)/(individual[4]+1),
    black = [1, 2, 2, 2, 8]
    board = chess.Board(make_board_fen(black, individual))
    wins = 0

    for _ in range(3):
        outcome = await play(
            engine,
            board,
            selfplay=True,
            pvs=1,
            time_limit=chess.engine.Limit(time=1),
            debug=logging.basicConfig(level=logging.DEBUG),
            printout=False
        )
        
    print(outcome)
    if outcome:
        wins += 1

    return wins,  # Return a tuple

# Define the problem as a maximization problem
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


async def main():
    engine = await load_engine_from_cmd("python sunfish.py")
    
       
    try:
        result = await run_ea(engine)
        # Process the result as needed
        print("Best individual:", result)
        print("Fitness of best individual:", result.fitness.values)
        
        black = [1, 2, 2, 2, 8]
        board = chess.Board(make_board_fen(black, result))
        wins = 0

        for _ in range(3):
            outcome = await play(
                engine,
                board,
                selfplay=True,
                pvs=1,
                time_limit=chess.engine.Limit(time=1),
                debug=True,
                printout=False
            )
        
            print(board)
            print(outcome)
            if outcome:
                wins += 1

        print(wins),  # Return a tuple
    finally:
        await engine.quit()


asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
