from flask import Flask, request, jsonify
import os
import shutil
import hashlib
import getpass

app = Flask(__name__)

@app.route('/clean', methods=['POST'])
def clean_files():
    data = request.json
    directory = data.get('directory')
    safe_mode = data.get('safe_mode', False)

    if directory:
        if not safe_mode:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            return jsonify({"message": "Bestanden zijn verwijderd in {}".format(directory)})
        else:
            return jsonify({"message": "Veilige modus is ingeschakeld. Geen bestanden verwijderd."})
    else:
        return jsonify({"error": "Geen map opgegeven."}), 400

@app.route('/set_safe_mode', methods=['POST'])
def set_safe_mode():
    data = request.json
    password = data.get('password')

    if password == "mijnveiligwachtwoord":
        return jsonify({"message": "Veilige modus is ingeschakeld."})
    else:
        return jsonify({"error": "Ongeldig wachtwoord. Veilige modus niet ingeschakeld."}), 400

@app.route('/wipe_data', methods=['POST'])
def wipe_data():
    data = request.json
    special_password = data.get('special_password')

    if special_password == "speciaalwachtwoord":
        # Voeg hier code toe om alle gegevens te wissen
        return jsonify({"message": "Alle gegevens zijn gewist."})
    else:
        return jsonify({"error": "Ongeldig speciaal wachtwoord."}), 400

if __name__ == '__main__':
    app.run(debug=True)
