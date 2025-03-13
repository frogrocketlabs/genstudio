import genstudio.plot as Plot
import numpy as np

Plot.bitmap(np.random.rand(8, 8)).save_pdf("scratch/bitmap.pdf", debug=True)
