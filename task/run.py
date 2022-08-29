from psychopy import visual, core, event
from psychtoolbox import WaitSecs
from events import EventMarker
from functions import *

FREQS = [130, 200, 280]
SEQ_LENS = [30, 36, 42]
TONE_LEN = 0.3
SUB_NUM = input("Input subject number: ")
BLOCK_NUM = input("Input block number: ")
WIN = visual.Window(size = (800, 600),
    screen = -1,
    units = "norm",
    fullscr = False,
    pos = (0, 0),
    allowGUI = False)
    

KB = get_keyboard('Dell Dell USB Entry Keyboard')
MARKER = EventMarker()

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



#have subj listen to 3 pitches
#display instructions if training block
#welcome to block
if BLOCK_NUM == "0": 
    SCORE_NEEDED = 3
    hear_pitches(WIN, TONE_LEN, FREQS)
    instructions(WIN)
else:
    event.clearEvents(eventType = None)
    blk_welcome = visual.TextStim(WIN, 
                                  text = f"Welcome to block number {BLOCK_NUM}. Press 'space' to continue.",
                                  pos=(0.0, 0.0),
                                  color=(1, 1, 1), 
                                  colorSpace='rgb' )
    blk_welcome.draw()
    WIN.flip()
    event.waitKeys(keyList = ['space'])
    SCORE_NEEDED =18


# play sequences until SCORE_NEEDED is reached
while score < SCORE_NEEDED:
    target = get_target(FREQS)
    n_tones = get_n_tones(SEQ_LENS)

    # Play target
    play_target(WIN, TONE_LEN, target)
    ready(WIN)
    WaitSecs(0.5)

    # Play tones
    fixation(WIN)
    WaitSecs(1)
    tone_nums, freqs, marks, is_targets, n_targets = play_sequence(MARKER, FREQS, TONE_LEN, target, n_tones)
    WIN.flip()
    WaitSecs(0.5)

    # Get response
    response = get_response(WIN)
    print(f'n_targets: {n_targets}')
    print(f'response: {response}')
    correct, score = update_score(WIN, n_targets, response, score, SCORE_NEEDED)
    print(f'score: {score}')
    seq_num += 1
    print(f'seq_num: {seq_num}')

    # Write log file
    write_log(LOG, n_tones, SEED, SUB_NUM, BLOCK_NUM, seq_num, target, tone_nums,
              freqs, marks, is_targets, n_targets, response, correct, score)
    WaitSecs(1)

print("Block over.")

core.quit()
