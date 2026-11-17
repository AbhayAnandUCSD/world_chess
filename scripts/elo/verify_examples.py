"""Recheck selected gains and losses at depth 18, comparing the two candidate moves."""
import json
import sys
import chess
import chess.engine
from wc.model import ROOT

D=ROOT/"results/18"
positions={r["gid"]+"|"+str(r["ply"]):r for r in map(json.loads,(D/"positions.jsonl").open())}
examples=json.loads((D/"examples.json").read_text())
with chess.engine.SimpleEngine.popen_uci(sys.argv[1]) as engine:
    engine.configure({"Threads":1,"Hash":128})
    for example in examples:
        row=positions[example["gid"]+"|"+str(example["ply"])]
        board=chess.Board(row["fen"])
        moves=[board.parse_san(example[key]) for key in ["base_move","copy_move"]]
        lines=engine.analyse(board,chess.engine.Limit(depth=18),root_moves=moves,multipv=2)
        evaluations={}
        for line in lines:
            san=board.san(line["pv"][0])
            score=line["score"].pov(board.turn)
            evaluations[san]={"centipawns":score.score(),"mate":score.mate()}
        example["depth18"]=evaluations
        example["side"]="White" if board.turn else "Black"
        example["descriptions"]={}
        for key,move in zip(["base_move","copy_move"],moves):
            piece=chess.piece_name(board.piece_at(move.from_square).piece_type)
            example["descriptions"][key]=f"{piece} from {chess.square_name(move.from_square).upper()} to {chess.square_name(move.to_square).upper()}"
(D/"examples_verified.json").write_text(json.dumps(examples,indent=2))
