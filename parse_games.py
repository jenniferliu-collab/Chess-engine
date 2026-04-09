import io
import chess
import chess.pgn
import zstandard as zstd
import requests
import numpy as np

# 1. Downloading and streaming a Lichess PGN file
# Because the files are so large, we stream them instead of downloading the whole thing.

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"
MAX_GAMES = 500


def stream_pgn_from_lichess(url: str):
    """
    Get one decoded text chunk at a time from the Lichess URL
    Uses streaming so we don't load the whole file into memory.
    """
    dctx = zstd.ZstdDecompressor()
    
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        
        # zstd needs a file-like object — wrap the raw stream
        with dctx.stream_reader(response.raw) as reader:
            # chess.pgn.read_game() needs a text stream, not bytes
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            yield text_stream  # yield the whole stream object

# 2. Encode a board position as a flat numpy vector
# We represent the board as 12 binary planes of 8x8:
#   Planes 0-5:  white pieces (P, N, B, R, Q, K)
#   Planes 6-11: black pieces (p, n, b, r, q, k)
# A 1 means that piece is on that square, 0 means it isn't.
# Flattened: 12 * 64 = 768 numbers total.

PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

def board_to_vector(board: chess.Board) -> np.ndarray:
    """
    Convert a chess.Board into a float32 vector with 768 numbers total to represent entire chessboard
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    
    for square, piece in board.piece_map().items():
        plane_idx = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        row = square // 8   # rank 0-7
        col = square % 8    # file 0-7
        planes[plane_idx, row, col] = 1.0
    
    return planes.flatten()  # shape: (768,)


# 3. Parse game result into a label 
# PGN result strings: "1-0" (white wins), "0-1" (black wins), "1/2-1/2" (draw)

def result_to_label(result: str):
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    elif result == "1/2-1/2":
        return 0.0
    return None  # "*" means game was abandoned — skip it


# 4. Build the dataset 
# Strategy: for each game, we sample positions at several points but not for every single move
# We label each position with the final game result.

def build_dataset(url: str, max_games: int = 500, sample_every: int = 5):
    """
    Returns:
        positions: np.ndarray of shape (N, 768)
        labels:    np.ndarray of shape (N,)
    """
    positions = []
    labels = []
    games_parsed = 0
    
    for text_stream in stream_pgn_from_lichess(url):
        while games_parsed < max_games:
            # chess.pgn.read_game reads one game at a time from the stream
            game = chess.pgn.read_game(text_stream)
            
            if game is None:
                break  
            
            result = game.headers.get("Result", "*")
            label = result_to_label(result)
            
            if label is None:
                continue  # skip abandoned games
            
            # Walk through moves, sample positions periodically
            board = game.board()
            move_number = 0
            
            for move in game.mainline_moves():
                board.push(move)
                move_number += 1
                
                # Sample every Nth position to get variety
                if move_number % sample_every == 0:
                    positions.append(board_to_vector(board))
                    labels.append(label)
            
            games_parsed += 1
            if games_parsed % 50 == 0:
                print(f"Parsed {games_parsed} games, {len(positions)} positions so far...")
    
    return np.array(positions), np.array(labels)


# 5. Run it 

if __name__ == "__main__":
    print("Downloading and parsing games...")
    X, y = build_dataset(URL, max_games=MAX_GAMES)
    
    print(f"\nDataset ready:")
    print(f"  Positions shape: {X.shape}")   # e.g. (2500, 768)
    print(f"  Labels shape:    {y.shape}")   # e.g. (2500,)
    print(f"  Label breakdown: white={np.sum(y==1)}, draw={np.sum(y==0)}, black={np.sum(y==-1)}")
    
    # Save for later use in training
    np.save("positions.npy", X)
    np.save("labels.npy", y)
    print("\nSaved to positions.npy and labels.npy")