from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

drawing = svg2rlg("episode_reward.svg")
renderPDF.drawToFile(drawing, "file1.pdf")
