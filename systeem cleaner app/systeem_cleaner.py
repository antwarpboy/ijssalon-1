import os
import shutil
import hashlib
import getpass
import tkinter as tk
from tkinter import filedialog

class geavanceerdsysteemcleaner:
    def __init__(self, master):
        self.master = master
        self.master.title("Advanced Cleaner")

        self.safe_mode = False
        self.password = None

        self.selected_folders = []

        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self.master, text="Selecteer een map om schoon te maken:")
        self.label.pack()

        self.browse_button = tk.Button(self.master, text="Bladeren", command=self.browse_directory)
        self.browse_button.pack()

        self.folder_listbox = tk.Listbox(self.master, selectmode=tk.MULTIPLE, height=5)
        self.folder_listbox.pack()

        self.clean_button = tk.Button(self.master, text="Schoonmaken", command=self.clean_files)
        self.clean_button.pack()

        self.safe_mode_button = tk.Button(self.master, text="Veilige modus", command=self.set_safe_mode)
        self.safe_mode_button.pack()

        self.secret_button = tk.Button(self.master, text="Speciaal wachtwoord", command=self.set_special_password)
        self.secret_button.pack()

    def browse_directory(self):
        directory = filedialog.askdirectory()
        self.directory_to_clean = directory
        self.label.config(text=f"Geselecteerde map: {directory}")

        self.folder_listbox.delete(0, tk.END)
        for root, dirs, files in os.walk(directory):
            for dir in dirs:
                self.folder_listbox.insert(tk.END, os.path.join(root, dir))

    def clean_files(self):
        if hasattr(self, 'directory_to_clean'):
            selected_indices = self.folder_listbox.curselection()
            self.selected_folders = [self.folder_listbox.get(idx) for idx in selected_indices]
            directory = self.directory_to_clean
            if not self.safe_mode:
                for folder in self.selected_folders:
                    shutil.rmtree(folder)
                    print(f"Map verwijderd: {folder}")
            else:
                print("Veilige modus is ingeschakeld. Geen mappen verwijderd.")
        else:
            print("Selecteer eerst een map om schoon te maken.")

    def encrypt_file(self, file_path):
        with open(file_path, 'rb') as f:
            data = f.read()
            encrypted_data = hashlib.sha256(data).hexdigest()
            return encrypted_data

    def set_safe_mode(self):
        password = getpass.getpass("Voer het veilige modus wachtwoord in: ")
        # Simpel voorbeeld van wachtwoordcontrole - in de praktijk zou je een sterkere methode gebruiken
        if password == self.password:
            self.safe_mode = True
            print("Veilige modus is ingeschakeld.")
        else:
            print("Ongeldig wachtwoord. Veilige modus niet ingeschakeld.")

    def set_special_password(self):
        self.password = getpass.getpass("Kies een speciaal wachtwoord om alle gegevens te wissen: ")

    def secret_function(self):
        if self.safe_mode:
            special_password = getpass.getpass("Voer het speciale wachtwoord in om alle gegevens te wissen: ")
            if special_password == self.password:
                print("Alle gegevens worden gewist...")
                # Voeg hier code toe om alle gegevens te wissen
                print("Alle gegevens zijn gewist.")
            else:
                print("Ongeldig speciaal wachtwoord.")
        else:
            print("Schakel eerst de veilige modus in.")

def main():
    root = tk.Tk()
    app = geavanceerdsysteemcleaner(root)
    root.mainloop()

if __name__ == "__main__":
    main()
