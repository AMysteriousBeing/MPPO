## Process Data Pipeline

1. get match id from botzone
2. download match info from botzone (both step 1 and step 2 can be done with get_match_id.py)
3. process all data to .txt file (through process_botzone_raw.py)
4. convert to individual .json file (through preprocess_pass1_optimized.py)
5. (optional) further calculate shanten number for each game state if necessary (through preprocess_pass2.py)