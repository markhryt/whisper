import os
import tempfile
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from groq import Client

# Load environment variables
load_dotenv()

# Get API key
groq_api_key = os.getenv("GROQ_KEY")

# Ensure the key is loaded
if not groq_api_key:
    raise ValueError("GROQ_KEY is not set in the environment variables")

# Initialize the Groq client
client = Client(api_key=groq_api_key)

# Initialize Flask app
app = Flask(__name__)

# Path to store audio files
AUDIO_FILE_PATH = "/Users/markhrytchak/Documents/projects/devpost/devfest25/whisper/audiofiles/"

def analyze_risk(transcription_text):
    """
    Analyzes transcribed text to determine if it signals distress.
    """
    danger_prompt = (
        """Analyze the following text and determine if it signals danger. 
        Since we are limited in our ability to recognize speech, consider words that may be phonetically 
        similar to distress calls. Example: If the text is 'Hello', it may mean 'Help'. 
        Respond with 'YES' if there is danger, otherwise respond with 'NO'. 
        Provide an explanation. Here is the text:\n\n""" + transcription_text
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": danger_prompt}]
        )

        chatbot_response = response.choices[0].message.content.strip()

        # Check if AI detected danger
        danger_detected = "YES" in chatbot_response.upper()
    
        return danger_detected, chatbot_response
    except Exception as e:
        return False, f"Error in risk analysis: {str(e)}"

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """
    Transcribes multiple audio files sequentially and checks if they signal distress.
    """
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio files uploaded"}), 400

        files = request.files.getlist("audio")  # Accept multiple files
        results = {}

        for file in files:
            file_path = os.path.join(AUDIO_FILE_PATH, file.filename)
            file.save(file_path)  # Save file to the directory

            # Create a temporary copy for processing
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, file.filename)
            os.system(f"cp {file_path} {temp_file_path}")

            try:
                # Transcribe the audio
                with open(temp_file_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        file=(file.filename, audio_file.read()),
                        model="whisper-large-v3-turbo",
                        response_format="json",
                        language="en",
                        temperature=0.0
                    )

                transcribed_text = transcription.text

                # Perform danger analysis immediately after transcription
                danger_detected, risk_analysis = analyze_risk(transcribed_text)

                # Store results for this file
                results[file.filename] = {
                    "transcription": transcribed_text,
                    "danger_detected": danger_detected,
                    "risk_analysis": risk_analysis
                }

            except Exception as e:
                results[file.filename] = {"error": str(e)}

            os.remove(temp_file_path)  # Cleanup temporary file

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)  # Enable debug mode
    