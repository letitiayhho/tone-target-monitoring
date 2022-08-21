#RUN_TRAINING
#TRAINING BLOCK

from psychopy import visual, core, event
from functions import *

FREQS = [130, 200, 280]
# SEQ_LENS = [30, 36, 42]
SEQ_LENS = [4, 6, 8]
TONE_LEN = 0.3
SCORE_NEEDED = 3
SUB_NUM = input("Input subject number: ")
BLOCK_NUM = "0"
WIN = visual.Window([800,600], monitor="testMonitor", units="deg")

#marker = EventMarker()

# open log file
LOG = open_log(SUB_NUM, BLOCK_NUM)
score = get_score(LOG)
print(f"score: {score}")
seq_num = get_seq_num(LOG)
print(f"seq_num: {seq_num}")

# set subject number, block and seq_num as seed
SEED = int(SUB_NUM + "0" + BLOCK_NUM + str(seq_num))
print("Current seed: " + str(SEED))
random.seed(SEED)

#Provide Detailed Task Instructions: 
instructions(WIN)


# play sequences until SCORE_NEEDED is reached
while score < SCORE_NEEDED:
    target = get_target(FREQS)
    n_tones = get_n_tones(SEQ_LENS)
    
    # Play target
    play_target(WIN, TONE_LEN, target)
    ready(WIN)
    core.wait(0.5)
    
    # Play tones
    fixation(WIN)
    core.wait(1)
    tone_nums, freqs, marks, is_targets, n_targets = play_sequence(FREQS, TONE_LEN, target, n_tones)
    WIN.flip()
    core.wait(0.5)
    
    # Get response
    response = get_response(WIN)
    print(f'response: {response}')
    correct, score = update_score(WIN, n_targets, response, score, SCORE_NEEDED)
    print(f'score: {score}')
    seq_num += 1
    print(f'seq_num: {seq_num}')
    
    # Write log file
    write_log(LOG, n_tones, SEED, SUB_NUM, BLOCK_NUM, seq_num, target, tone_nums, 
              freqs, marks, is_targets, n_targets, response, correct, score)
    core.wait(1)

core.quit()