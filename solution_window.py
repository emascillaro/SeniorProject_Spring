import tkinter as tk
from Calc_Scripts import calculation

equationString = "25+247"
solution = calculation(equationString)
print(solution)

window = tk.Tk()
answer_label = tk.Label(text = solution)
answer_label.pack()

return_button = tk.Button(text = "Take another Photo", command = lambda: controller.show_frame(StartPage))
return_button.pack()

window.mainloop()