import numpy as np
import sys
import os
import glob
import matplotlib.pyplot as plt
import metrics
import tkinter


def plot(listbox, event):
    selected_files = [listbox.get(i) for i in listbox.curselection()]
    max_loc = None

    for name in selected_files:
        data = np.genfromtxt(name, delimiter=',', skip_header=1)
        y_pred = data[:, 1]
        y_true = data[:, 0]
        label = '{0} MAE:{1:.2f}'.format(os.path.splitext(os.path.basename(name))[0], metrics.mae(y_true, y_pred))

        if max_loc is None:
            max_loc = np.argmax(y_true)
            plt.plot(y_true, label='True', color='black')
            plt.plot(y_pred, label=label)
        else:
            offset = np.argmax(y_true) - max_loc
            plt.plot(np.arange(len(y_pred)) - offset, y_pred, label=label)
    plt.legend()
    plt.grid()
    plt.show()


def on_path_changed(var, listbox, e):
    listbox.delete(0, tkinter.END)
    filenames = glob.glob(var.get())
    for name in filenames:
        listbox.insert(tkinter.END, name)


def main():
    filenames = glob.glob(sys.argv[1])

    root = tkinter.Tk()
    root.minsize(width=400, height=500)
    scrollbar = tkinter.Scrollbar(root)
    pathvar = tkinter.StringVar()
    pathvar.set(sys.argv[1])
    textbox = tkinter.Entry(root, textvariable=pathvar)
    listbox = tkinter.Listbox(root, height=5, width=15, selectmode=tkinter.EXTENDED)
    button = tkinter.Button(root, text="Plot")
    for name in filenames:
        listbox.insert(tkinter.END, name)
    textbox.pack(fill=tkinter.X, expand=tkinter.NO, padx=5, pady=5)
    scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)
    listbox.pack(fill=tkinter.BOTH, expand=tkinter.YES, padx=5, pady=5)
    button.pack(fill=tkinter.X, expand=tkinter.NO, padx=5, pady=5)

    textbox.bind('<Return>', lambda e: on_path_changed(pathvar, listbox, e))
    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    button.bind('<Button-1>', lambda e: plot(listbox, e))
    root.mainloop()


if __name__ == '__main__':
    main()
