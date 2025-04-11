from flask import Flask, request, jsonify, render_template
import torch
import os

from model import Seq2SeqModel, Encoder, Decoder, AttentionMechanism, beam_search
from tokenizer import SimpleTokenizer

app = Flask(__name__)

MODEL_DIR = "models"
MODEL_FILENAME = "chatbot_model.pt"  # The actual model file name
VOCAB_PATH = "processed_dailydialog/vocab.txt"  # The vocabulary file for the tokenizer

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = None
tokenizer = None

try:
    # Load the tokenizer
    print(f"Loading tokenizer vocabulary from: {VOCAB_PATH}")
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Vocabulary file not found at {VOCAB_PATH}")
    
    tokenizer = SimpleTokenizer()
    tokenizer.load_vocab(VOCAB_PATH)
    print("Tokenizer loaded successfully.")
    
    # Get vocabulary sizes for model initialization
    input_dim = output_dim = len(tokenizer.word2idx)
    
    # Load the model
    print(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    # Initialize model components with the same architecture as during training
    embedding_dim = 64  
    hidden_dim = 256    
    
    # Create model components
    encoder = Encoder(input_dim, embedding_dim, hidden_dim)
    attention = AttentionMechanism(hidden_dim)
    decoder = Decoder(output_dim, embedding_dim, hidden_dim, attention)
    
    # Create the Seq2Seq model
    model = Seq2SeqModel(encoder, decoder, device)
    
    # Load the saved model weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading model or tokenizer: {e}. Please ensure paths and filenames are correct in app.py.")
    exit()
except ImportError as e:
    print(f"Import error: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    exit()

# Route for HTML Interface
@app.route('/')
def index():
    # Serves the index.html file
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not model or not tokenizer:
         return jsonify({"error": "Model or Tokenizer not loaded. Check server logs."}), 500

    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Invalid request: 'message' field missing"}), 400

        user_message = data['message'].strip()
        print(f"Received message: {user_message}")

        if not user_message:
            return jsonify({"response": ""}) # Return empty if user sends whitespace

        print("Processing input...")
        # Convert text to token indices
        input_indices = tokenizer.encode(user_message)
        # Convert to tensor and move to device
        input_tensor = torch.tensor([input_indices], device=device)
        
        print(f"Generating response using {device}...")
        with torch.no_grad():
            # Ensure model is on the correct device
            model.to(device)
            
            # Use beam search for better quality responses
            bot_response = beam_search(
                model=model,
                tokenizer=tokenizer,
                input_seq=input_tensor,
                beam_width=3,
                max_length=30
            )
        
        print(f"Generated response: {bot_response}")
        return jsonify({"response": bot_response})

    except Exception as e:
        print(f"Error processing chat request: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
