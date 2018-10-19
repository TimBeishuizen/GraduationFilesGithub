from tkinter import *

from tkinter.ttk import *
from cBioFTesting import read_csv_dataset

class GUI_cBioF(Tk):
    """

    cBioF GUI

    """

    X = None
    y = None
    features = None

    def __init__(self, *args):

        super().__init__(*args)

        self.title("cBioF")

        self.geometry('350x200')

        # Read TAB
        self.tabControl = Notebook(self)  # Create Tab Control

        self.add_read_tab()
        self.add_explore_tab()
        self.add_anayse_tab()

        self.tabControl.pack(expand=1, fill="both")  # Pack to make visible


    def add_read_tab(self):
        # Add buttons etc. in read tab
        self.read_tab = Frame(self.tabControl)  # Create a tab
        self.tabControl.add(self.read_tab, text='Read in datsaet')  # Add the tab

        self.entry_label = Label(self.read_tab, text="Directory and filename CSV file:")
        self.entry_label.place(x=0, y=5)

        self.dataset_name = Entry(self.read_tab)
        self.dataset_name.place(x=0, y=30)
        self.dataset_name.insert(END, 'FILENAME.csv')

        self.read_dataset = Button(self.read_tab, text="Read CSV file", command=self.read_input)
        self.read_dataset.place(x=0, y=55)

    def add_explore_tab(self):
        # Add buttons etc. in explore tab
        self.expl_tab = Frame(self.tabControl)  # Create a tab
        self.tabControl.add(self.expl_tab, text='Explore dataset')  # Add the tab

        self.prep_expl = Checkbutton(self.expl_tab, text='Preprocessing')
        self.prep_expl.place(x=0, y=5)

        self.class_expl = Checkbutton(self.expl_tab, text='Classification')
        self.class_expl.place(x=0, y=30)

        self.missing_label = Label(self.expl_tab, text='Missing values:')
        self.missing_label.place(x=0, y=55)

        self.missing_values = Entry(self.expl_tab)
        self.missing_values.place(x=100, y=55)
        self.missing_values.insert(END, 'Unknown')

        self.explore_dataset = Button(self.expl_tab, text='Explore dataset', command=self.explore_input)
        self.explore_dataset.place(x=0, y=80)

    def add_anayse_tab(self):

        self.ana_tab = Frame(self.tabControl)  # Create a tab
        self.tabControl.add(self.ana_tab, text='Analyse dataset')  # Add the tab

        self.prep_ana = Checkbutton(self.ana_tab, text='Preprocessing')
        self.prep_ana.place(x=0, y=5)

        self.class_ana = Checkbutton(self.ana_tab, text='Classification')
        self.class_ana.place(x=0, y=30)

        self.fs_ana = Checkbutton(self.ana_tab, text='Feature selection')
        self.fs_ana.place(x=0, y=55)

        self.file_label = Label(self.ana_tab, text="File name for pipeline file:")
        self.file_label.place(x=0, y=80)

        self.file_entry = Entry(self.ana_tab)
        self.file_entry.place(x=0, y=105)
        self.file_entry.insert(END, 'No output file')

        self.analyse_dataset = Button(self.ana_tab, text='Explore dataset', command=self.analyse_input)
        self.analyse_dataset.place(x=0, y=130)

    def read_input(self):
        input = self.dataset_name.get()
        try:
            self.X, self.y, self.features = read_csv_dataset.read_csv_dataset(input)
            self.print_progress(self.read_tab, "Read and imported the file %s" % input)
        except:
            self.print_progress(self.read_tab, "Invalid file name or does not exist: %s" % input)

    def explore_input(self):
        print('TO BE DONE')

    def analyse_input(self):
        print('TO BE DONE')

    def print_progress(self, print_location, print_statement):
        print_label = Label(print_location, text=print_statement)
        print_label.place(x=0, y=160)


GUI = GUI_cBioF()
GUI.mainloop()