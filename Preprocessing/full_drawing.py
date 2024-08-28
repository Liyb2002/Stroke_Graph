import proc_CAD.draw_all_lines
import os

data_idx = 0
dir = 'testOnly'
data_directory = os.path.join(dir, f'data_{data_idx}')
proc_CAD.draw_all_lines.run(data_directory)